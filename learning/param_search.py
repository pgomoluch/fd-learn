#!/usr/bin/python3

import ast
import glob
import numpy as np
import os
import random
import subprocess
import sys
import time

from rl_common import get_output_with_timeout, get_cost, compute_reference_costs

PARAMS_PATH = '../params.txt'
HEURISTIC = 'h1=ff(transform=adapt_costs(one))'
SEARCH = 'parametrized(h1,params=%s)'
CONDOR_DIR = 'condor'
THREADS = 4

# Random walk optimization
INITIAL_PARAMS = [0.2, 5, 0.5, 20]
UNITS = [0.05, 1, 0.05, 5]

# Cross-entropy Method
# [epsilon, stall_size, number of random walks, random_walk_length, cycle_length, fraction_local]
INITIAL_MEAN = [0.5, 10.0, 5.0, 10.0, 200.0, 0.5]
INITIAL_STDDEV = [0.5, 10.0, 5.0, 10.0, 200.0, 0.5]

MIN_PARAMS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
MAX_PARAMS = [1.0, float('inf'), float('inf'), float('inf'), float('inf'), 1.0]

TARGET_TYPES = [float, int, int, int, int, float]


POPULATION_SIZE = 120
ELITE_SIZE = 10
N_TEST_PROBLEMS = 50
RUNS_PER_PROBLEM = 4
MAX_PROBLEM_TIME = 5.0


def generate_next(params):
    param_id = random.randint(0, len(params)-1)
    while True:
        sign = random.randint(0,1)
        value = params[param_id]
        if sign:
            value += UNITS[param_id]
        else:
            value -= UNITS[param_id]
        if value >= MIN_PARAMS[param_id] and value <= MAX_PARAMS[param_id]: 
            result = params.copy()
            result[param_id] = value
            return result

def save_params(params, filename):
    f = open(filename, 'w')
    for (p, t) in zip(params, TARGET_TYPES):
        if t == int:
            p = int(round(p))
        f.write(str(p))
        f.write('\n')
    f.close()

def bound_params(params):
    for i in range(len(params)):
        if params[i] < MIN_PARAMS[i]:
            params[i] = MIN_PARAMS[i]
        elif params[i] > MAX_PARAMS[i]:
            params[i] = MAX_PARAMS[i]

def get_problem():
    if get_problem.problem_set is None:
        problems = glob.glob(PROBLEM_DIR + '/p*.pddl')
        cost_dict_path = os.path.join(PROBLEM_DIR, 'costs.npy')
        if os.path.exists(cost_dict_path):
            # Old cost format: a single dictionary of minimum costs 
            costs = np.load(cost_dict_path).item()
            get_problem.problem_set = [ (p, costs[p.split('/')[-1]]) for p in problems ]
        else:
            # New cost format: per-problem dictionaries of costs obtained
            # with different configurations.
            get_problem.problem_set = []
            for p in problems:
                costs_path = p.replace('.pddl', 'costs.txt')
                costs = ast.literal_eval(open(costs_path).read())
                costs = list(filter(lambda x: x >= 0, costs.values()))
                if len(costs) > 0:
                    c = min(costs)
                else:
                    c = -1
                get_problem.problem_set.append((p,c))
                
    problem_and_cost = random.choice(get_problem.problem_set)
    print(problem_and_cost)
    return problem_and_cost

get_problem.problem_set = None
get_problem.costs = None

def compute_ipc_reward(plan_cost, reference_cost):
    # If no reference solution
    if reference_cost == -1:
        if plan_cost > 0:
            return 2.0
        raise NoRewardException()
    # Otherwise
    if plan_cost < 0:
        return 0.0
    elif plan_cost == 0:
        raise NoRewardException()
    else:
        return reference_cost / plan_cost

def score_params(params, paths_and_costs):
    save_params(params, PARAMS_PATH)
    total_score = 0.0
    for problem_path, reference_cost in paths_and_costs:
        for i in range(RUNS_PER_PROBLEM):
            plan_cost = -1
            try:
                problem_start = time.time()
                planner_output = get_output_with_timeout(['../fast-downward.py',
                    '--build', 'release64',  DOMAIN,  problem_path,
                    '--heuristic', HEURISTIC, '--search',  SEARCH % PARAMS_PATH],
                    MAX_PROBLEM_TIME)
                problem_time = time.time() - problem_start
                print('Problem time: ', problem_time)
                plan_cost = get_cost(planner_output)
            except subprocess.TimeoutExpired:
                print('Failed to find a solution.')
                problem_time = -1
            
            try:
                reward = compute_ipc_reward(plan_cost, reference_cost)
            except:
                reward = 0.0
                
            total_score += reward
    
    return total_score


CONDOR_SCRIPT = """#!/bin/bash
cd condor/$1
i=0
while read problem; do
    cd $i
    /usr/bin/python {fd_path} --build release64 --overall-time-limit {time_limit} {domain} $problem --heuristic "{heuristic}" --search "{search}" > fd.out
    cd ..
    ((i++))
done < problems.txt
"""

CONDOR_PAR_SCRIPT = """#!/bin/bash
start=$(($1*{threads}))
for ((i=0; i<{threads}; i++))
do
    {condor_script} $((start+i)) &
    pid[$i]=$!
done

for p in ${{pid[*]}}
do
    wait $p
done
"""

CONDOR_SPEC = """universe = vanilla
executable = {condor_script}
output = condor/$(Process)/condor.out
error = condor/$(Process)/condor.err
log = condor/condor.log
arguments = $(Process)
requirements = regexp("^sprite[0-9][0-9]", TARGET.Machine)
queue {population_size}
"""

def get_condor_cluster(submit_output):
    #TODO find a better way to get this number
    space = submit_output.rfind(' ')
    dot = submit_output.rfind('.')
    return int(submit_output[space+1 : dot])

def setup_condor():
    if not os.path.exists(CONDOR_DIR):
        os.makedirs(CONDOR_DIR)
    for h in range (POPULATION_SIZE):
        for i in range(N_TEST_PROBLEMS):
            path = os.path.join(CONDOR_DIR, str(h), str(i))
            if not os.path.exists(path):
                os.makedirs(path)
    # Condor script
    condor_script_path = os.path.join(CONDOR_DIR, 'condor.sh')
    condor_file = open(condor_script_path, 'w')
    condor_file.write(CONDOR_SCRIPT.format(
        fd_path=os.path.abspath('../fast-downward.py'),
        time_limit=int(MAX_PROBLEM_TIME),
        domain=os.path.abspath(DOMAIN),
        heuristic=HEURISTIC,
        search=SEARCH % PARAMS_PATH))
    condor_file.close()
    # Condor parallel script
    condor_par_script_path = os.path.join(CONDOR_DIR, 'condor_par.sh')
    condor_par_file = open(condor_par_script_path, 'w')
    condor_par_file.write(CONDOR_PAR_SCRIPT.format(
        threads=THREADS,
        condor_script=os.path.join('.', CONDOR_DIR, 'condor.sh')))
    condor_par_file.close()
    
    os.chmod(condor_script_path, 0o775)
    os.chmod(condor_par_script_path, 0o775)

def condor_score_params(all_params, paths_and_costs, log=None):
    problem_list = os.linesep.join([os.path.abspath(p) for (p, _) in paths_and_costs]) + os.linesep
    for params_id, params in enumerate(all_params):
        save_params(params, os.path.join(CONDOR_DIR, str(params_id), 'params.txt'))
        problem_list_file = open(os.path.join(CONDOR_DIR, str(params_id), 'problems.txt'),'w')
        problem_list_file.write(problem_list)
        problem_list_file.close()
    
    condor_spec = CONDOR_SPEC.format(
        condor_script=os.path.abspath('condor/condor_par.sh'),
        population_size=POPULATION_SIZE/THREADS)
    condor_file = open('condor/condor.cmd', 'w')
    condor_file.write(condor_spec)
    condor_file.close()
    start_time = time.time()
    submit_output = subprocess.check_output(['condor_submit', 'condor/condor.cmd']).decode('utf-8')
    cluster_id = get_condor_cluster(submit_output)
    
    print('Waiting for condor...')
    #subprocess.check_call(['condor_wait', 'condor/condor.log'])
    # Discard the last 10% of jobs
    subprocess.check_call(['condor_wait', 'condor/condor.log', str(cluster_id), '-num', str(int(0.9 * POPULATION_SIZE/THREADS))])
    try:
        subprocess.check_call(['condor_rm', 'cluster', str(cluster_id)])
    except:
        print('condor_rm failed')
    elapsed_time = time.time() - start_time
    print('{} jobs ({} runs) completed in {} s.'.format(
        POPULATION_SIZE/THREADS, POPULATION_SIZE * N_TEST_PROBLEMS, round(elapsed_time,2)))
    if log:
        log.write(str(round(elapsed_time,2)) + '\n')
        log.flush()
    
    # Aggregate the results
    aggregation_start = time.time()
    total_scores = []
    for params_id in range(len(all_params)): # the same as POPULATION_SIZE
        total_score = 0.0
        for problem_id, (_, ref_cost) in enumerate(paths_and_costs):
            # Determine task completion in current iteration, based on sas_plan existence
            sas_plan_path = 'condor/{}/{}/sas_plan'.format(params_id,problem_id)
            if not os.path.exists(sas_plan_path):
                continue
            os.remove(sas_plan_path)
            
            output_path = 'condor/{}/{}/fd.out'.format(params_id,problem_id)
            output_file = open(output_path)
            planner_output = output_file.read()
            output_file.close()
            
            plan_cost = -1
            try:
                plan_cost = get_cost(planner_output)
            except:
                pass
            try:
                reward = compute_ipc_reward(plan_cost, ref_cost)
            except:
                reward = 0.0
            total_score += reward
        total_scores.append(total_score)
    
    print('Aggregation of the results took {} s.'.format(round(time.time()-aggregation_start)))
    return total_scores


DOMAIN = sys.argv[1]
PROBLEM_DIR = sys.argv[2]
TRAINING_TIME = int(sys.argv[3])

start_time = time.time()
params_log = open('params_log.txt', 'w')
condor_log = open('condor_log.txt', 'w')

setup_condor()

mean = np.array(INITIAL_MEAN)
cov = np.diag(np.square(INITIAL_STDDEV))

np.set_printoptions(suppress=True,precision=4)

while time.time() - start_time < TRAINING_TIME:
    
    print('Mean: ', mean)
    print('Covariance:\n', cov)
    
    # Choose the test problems
    paths_and_costs = []
    for i in range(N_TEST_PROBLEMS):
        paths_and_costs.append(get_problem())
    
    # Generate parameters
    params = np.random.multivariate_normal(mean, cov, POPULATION_SIZE)
    for row in params:
        bound_params(row)
    # For debugging: fix the parameters
    #params = np.concatenate((
    #    np.tile(np.array([[0.2,0,1000,0.0]]), (POPULATION_SIZE//2, 1)),
    #    np.tile(np.array([[0.2,20,100,0.9]]), (POPULATION_SIZE//2, 1))),
    #    axis=0)

    scores = condor_score_params(params, paths_and_costs, condor_log)
    
    scores = np.array(scores)
    sorted_ids = np.argsort(-scores)
    best_ids = sorted_ids[:ELITE_SIZE]
    mean = np.mean(params[best_ids], 0)
    cov = np.cov(params[best_ids], rowvar=False)

    condor_log.write('Best parameters:\n')
    for i in sorted_ids:
        condor_log.write(str(params[i]) + ' ' + str(scores[i]) + ' ' + str(i) + '\n')
    condor_log.write('\n')
    condor_log.flush()
    params_log.write(str(mean) + '\n' + str(cov) + '\n\n')
    params_log.flush()
    
    save_params(mean, 'params.txt')
    
params_log.close()
condor_log.close()


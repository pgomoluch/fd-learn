#!/usr/bin/python3

import glob
import numpy as np
import os
import random
import subprocess
import sys
import time

from rl_common import get_output_with_timeout, get_cost, compute_reference_costs

PARAMS_PATH = 'params.txt'
HEURISTIC = 'h1=ff(transform=adapt_costs(one))'
SEARCH = 'parametrized(h1,params=%s)'
CONDOR_DIR = 'condor'

MIN_PARAMS = [0.0, 0, 0.0, 1]
MAX_PARAMS = [1.0, float('inf'), 1.0, float('inf')]

INITIAL_PARAMS = [0.2, 5, 0.5, 20]
UNITS = [0.05, 1, 0.05, 5]

N_TEST_PROBLEMS = 10
RUNS_PER_PROBLEM = 5
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
    for p in params:
        f.write(str(p))
        f.write('\n')
    f.close()

def get_problem():
    if get_problem.problem_set is None:
        get_problem.problem_set = glob.glob(PROBLEM_DIR + '/p*.pddl')
        get_problem.costs = np.load(PROBLEM_DIR + '/costs.npy').item()
    problem = random.choice(get_problem.problem_set)
    print(problem)
    return (problem, get_problem.costs[problem.split('/')[-1]])

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
cd condor/$1/$2
/usr/bin/python {fd_path} --build release64 --overall-time-limit {time_limit} $3 $4 --heuristic "{heuristic}" --search "{search}"
"""

CONDOR_SPEC = """universe = vanilla
executable = {condor_script}
output = condor/{problem_id}/$(Process)/fd.out
error = condor/{problem_id}/$(Process)/fd.err
log = condor/condor.log
arguments = {problem_id} $(Process) {domain} {problem}
queue {runs_per_problem}
"""

def setup_condor():
    if not os.path.exists(CONDOR_DIR):
        os.makedirs(CONDOR_DIR)
    for i in range(N_TEST_PROBLEMS):
        for j in range(RUNS_PER_PROBLEM):
            path = os.path.join(CONDOR_DIR, str(i), str(j))
            if not os.path.exists(path):
                os.makedirs(path)
    condor_script_path = os.path.join(CONDOR_DIR, 'condor.sh')
    condor_file = open(condor_script_path, 'w')
    condor_file.write(CONDOR_SCRIPT.format(
        fd_path=os.path.abspath('../fast-downward.py'),
        time_limit=int(MAX_PROBLEM_TIME),
        heuristic=HEURISTIC,
        search=SEARCH % os.path.abspath(PARAMS_PATH)))
    condor_file.close()
    os.chmod(condor_script_path, 0o775)

def condor_score_params(params, paths_and_costs, log=None):
    save_params(params, PARAMS_PATH)
    
    # Run len(paths_and_costs) * RUNS_PER_PROBLEM condor jobs
    for problem_id, (problem_path, _) in enumerate(paths_and_costs):
        condor_spec = CONDOR_SPEC.format(
            condor_script=os.path.abspath('condor/condor.sh'),
            problem_id=problem_id,
            problem=os.path.abspath(problem_path),
            domain=os.path.abspath(DOMAIN),
            runs_per_problem=RUNS_PER_PROBLEM)
        condor_file = open('condor/condor.cmd', 'w')
        condor_file.write(condor_spec)
        condor_file.close()
        subprocess.check_call(['condor_submit', 'condor/condor.cmd'])
    start_time = time.time()
    print('Waiting for condor...')
    subprocess.check_call(['condor_wait', 'condor/condor.log'])
    elapsed_time = time.time() - start_time
    print(len(paths_and_costs) * RUNS_PER_PROBLEM, 'jobs completed in',
        str(round(elapsed_time,2)), 's.')
    if log:
        log.write(str(round(elapsed_time,2)) + '\n')
    
    # Aggregate the results
    total_score = 0.0
    for problem_id, (_, ref_cost) in enumerate(paths_and_costs):
        for attempt in range(RUNS_PER_PROBLEM):
            output_file = open('condor/{}/{}/fd.out'.format(problem_id,attempt))
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
    
    return total_score



DOMAIN = sys.argv[1]
PROBLEM_DIR = sys.argv[2]
TRAINING_TIME = int(sys.argv[3])

start_time = time.time()
params = INITIAL_PARAMS
params_log = open('params_log.txt', 'w')
condor_log = open('condor_log.txt', 'w')

setup_condor()

while time.time() - start_time < TRAINING_TIME:
    
    paths_and_costs = []
    for i in range(N_TEST_PROBLEMS):
        paths_and_costs.append(get_problem())
    new_params = generate_next(params)
    score = condor_score_params(params, paths_and_costs, condor_log)
    new_score = condor_score_params(new_params, paths_and_costs)
    
    print(params, score)
    print(new_params, new_score)
    if new_score > score:
        print('Change accepted\n')
        params = new_params

    params_log.write(str(params) + '\n')
    
params_log.close()
condor_log.close()


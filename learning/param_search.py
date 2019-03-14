#!/usr/bin/python3

import ast
import glob
import numpy as np
import os
import random
import subprocess
import sys
import time

from rl_common import get_output_with_timeout, get_cost, compute_reference_costs, compute_ipc_reward, save_params
from condor_evaluator import CondorEvaluator

from problem_generators.transport_generator import TransportGenerator
from problem_generators.parking_generator import ParkingGenerator
from problem_generators.elevators_generator import ElevatorsGenerator
from problem_generators.nomystery_generator import NomysteryGenerator

HEURISTIC = 'h1=ff(transform=adapt_costs(one))'
SEARCH = 'parametrized(h1,params=%s)'

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
N_TEST_PROBLEMS = 4
RUNS_PER_PROBLEM = 4
MAX_PROBLEM_TIME = 1800.0

STATE_FILE_PATH = 'search_state.npz'
GENERATE_PROBLEMS = True

if GENERATE_PROBLEMS:
    generator = TransportGenerator(4, 11, 30)
    difficulty_level = -7
    # generator = ParkingGenerator(21, 40)
    # difficulty_level = 0
    # generator = ElevatorsGenerator(60, 40, 10, 4, 1)
    # difficulty_level = 0
    # generator = NomysteryGenerator(15, 15, 1.5)
    # difficulty_level = 0

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
                if os.path.exists(costs_path):
                    costs = ast.literal_eval(open(costs_path).read())
                    costs = list(filter(lambda x: x >= 0, costs.values()))
                    if len(costs) > 0:
                        c = min(costs)
                    else:
                        c = -1
                else:
                    c = -1
                get_problem.problem_set.append((p,c))
                
    problem_and_cost = random.choice(get_problem.problem_set)
    print(problem_and_cost)
    return problem_and_cost

get_problem.problem_set = None
get_problem.costs = None

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

continuing = False

DOMAIN = sys.argv[1]
PROBLEM_DIR = sys.argv[2]
TRAINING_TIME = int(sys.argv[3])
if len(sys.argv) > 4:
    continuing = True
    STATE_FILE_PATH = sys.argv[4]

start_time = time.time()
params_log = open('params_log.txt', 'w')
condor_log = open('condor_log.txt', 'w')

evaluator = CondorEvaluator(
    target_types = TARGET_TYPES,
    population_size = POPULATION_SIZE,
    n_test_problems = N_TEST_PROBLEMS,
    domain_path = DOMAIN,
    heuristic_str = HEURISTIC,
    search_str = SEARCH,
    max_problem_time = MAX_PROBLEM_TIME)

if continuing:
    state_file = open(STATE_FILE_PATH, 'rb')
    npz_content = np.load(state_file)
    mean = npz_content['mean']
    cov = npz_content['cov']
    state_file.close()
else:
    mean = np.array(INITIAL_MEAN)
    cov = np.diag(np.square(INITIAL_STDDEV))

np.set_printoptions(suppress=True,precision=4)

while time.time() - start_time < TRAINING_TIME:
    
    print('Mean: ', mean)
    print('Covariance:\n', cov)
    
    # Choose the test problems
    paths_and_costs = []
    for i in range(N_TEST_PROBLEMS):
        if GENERATE_PROBLEMS:
            problem_path = 'tmp-problems/p' + str(i) + '.pddl'
            generator.generate(problem_path)
            paths_and_costs.append((problem_path, -1))
        else:
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

    #scores = condor_score_params(params, paths_and_costs, condor_log)
    scores = evaluator.score_params(params, paths_and_costs, condor_log)
    
    scores = np.array(scores)
    sorted_ids = np.argsort(-scores)
    best_ids = sorted_ids[:ELITE_SIZE]
    
    # Only update the distribution if some problems have been solved
    if scores[sorted_ids[0]] > 0.001:
        mean = np.mean(params[best_ids], 0)
        cov = np.cov(params[best_ids], rowvar=False)

    condor_log.write('Best parameters:\n')
    for i in sorted_ids:
        condor_log.write(str(params[i]) + ' ' + str(scores[i]) + ' ' + str(i) + '\n')
    condor_log.write('\n')
    condor_log.flush()
    params_log.write(str(mean) + '\n' + str(cov) + '\n\n')
    params_log.flush()
    
    save_params(mean, TARGET_TYPES, 'params.txt')
    state_file = open(STATE_FILE_PATH, 'wb')
    np.savez(state_file, mean=mean, cov=cov)
    state_file.close()
    
    if GENERATE_PROBLEMS:
        # Adjust problem difficulty based on current scores
        if scores[sorted_ids[POPULATION_SIZE // 2]] < 0.001:
            print('Decreasing problem difficulty...')
            generator.easier()
            difficulty_level -= 1
        elif scores[sorted_ids[3 * POPULATION_SIZE // 4]] > 0.001 and difficulty_level < 0:
            print('Increasing problem difficulty...')
            generator.harder()
            difficulty_level += 1
        print(generator)
    
params_log.close()
condor_log.close()


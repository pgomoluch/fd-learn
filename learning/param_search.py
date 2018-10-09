#!/usr/bin/python3

import glob
import numpy as np
import random
import subprocess
import sys
import time

from rl_common import get_output_with_timeout, get_cost, compute_reference_costs

sys.path.append('problem-generators')
from transport_generator import TransportGenerator
from parking_generator import ParkingGenerator
from elevators_generator import ElevatorsGenerator
from nomystery_generator import NomysteryGenerator


PARAMS_PATH = 'params.txt'
HEURISTIC = 'h1=ff(transform=adapt_costs(one))'
SEARCH = 'parametrized(h1,params=%s)' % PARAMS_PATH

MIN_PARAMS = [0.0, 0, 0.0, 1]
MAX_PARAMS = [1.0, float('inf'), 1.0, float('inf')]

INITIAL_PARAMS = [0.2, 5, 0.5, 20]
UNITS = [0.05, 1, 0.05, 5]

RUNS_PER_ITERATION = 5
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

def score_params(params, problem_path, reference_cost):
    save_params(params, PARAMS_PATH)
    total_score = 0.0
    for i in range(RUNS_PER_ITERATION):
        plan_cost = -1
        try:
            problem_start = time.time()
            planner_output = get_output_with_timeout(['../fast-downward.py',
                '--build', 'release64',  DOMAIN,  problem_path,
                '--heuristic', HEURISTIC, '--search',  SEARCH],
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


DOMAIN = sys.argv[1]
PROBLEM_DIR = sys.argv[2]
TRAINING_TIME = int(sys.argv[3])

start_time = time.time()
params = INITIAL_PARAMS
params_log = open('params_log.txt', 'w')

while time.time() - start_time < TRAINING_TIME:
    
    problem_path, reference_cost = get_problem()
    new_params = generate_next(params)
    score = score_params(params, problem_path, reference_cost)
    new_score = score_params(new_params, problem_path, reference_cost)
    
    print(params, score)
    print(new_params, new_score)
    if new_score > score:
        print('Change accepted\n')
        params = new_params

    params_log.write(str(params) + '\n')


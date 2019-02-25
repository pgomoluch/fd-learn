#!/usr/bin/python3

import glob
import os
import psutil
import random
import re
import signal
import subprocess
import sys
import time

import numpy as np

from pathlib import Path
from threading import Timer, Thread

from rl_common import *

from problem_generators.transport_generator import TransportGenerator
from problem_generators.parking_generator import ParkingGenerator
from problem_generators.elevators_generator import ElevatorsGenerator
from problem_generators.nomystery_generator import NomysteryGenerator


heuristic = 'h1=ff(transform=adapt_costs(one))'
search = 'learning(h1,t=%d)'
ref_search1 = 'eager_greedy(h1)'
ref_search2 = 'learning(h1,weights=weights2of6.txt)'
ref_search3 = 'learning(h1,weights=weights3of6.txt)'
ref_search4 = 'learning(h1,weights=weights4of6.txt)'
ref_search5 = 'learning(h1,weights=weights5of6.txt)'
ref_search6 = 'learning(h1,weights=weights6of6.txt)'
ref_search_list = [ref_search1, ref_search2, ref_search3,
    ref_search4, ref_search5, ref_search6]

learning_rate = 0.02
max_problem_time = 5.0
preprocessing_time = 800

STATE_SPACE = (2,2)
N_ACTIONS = 6

N_SAMPLES = 10
RUNS_PER_PROBLEM = 5

generator = TransportGenerator(4, 11)
#generator = ParkingGenerator(10, 18)
#generator = ElevatorsGenerator(20,12,6,2,2)
#generator = NomysteryGenerator(6,7,1.3)

if len(sys.argv) == 4:
    domain = sys.argv[1]
    problem_dir = sys.argv[2]
    training_time = int(sys.argv[3])
    generate = False
else:
    training_time = int(sys.argv[1])
    problem = 'problem.pddl'
    domain = '../../../IPC/own-transport/domain.pddl'
    #domain = '../../../IPC/own-parking/domain.pddl'
    #domain = '../../../IPC/own-elevators/domain.pddl'
    #domain = '../../../IPC/own-no-mystery/domain.pddl'
    generate = True


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def save_weights(weights):
    weights_file = open('weights.txt', 'w')
    for row in weights:
        weights_file.write(' '.join([str(x) for x in row.tolist()]))
        weights_file.write(' ')
    weights_file.close()

def n_states():
    r = 1
    for s in STATE_SPACE:
        r *= s
    return r

def get_problem():
    if generate:
        path = 'problem.pddl'
        generator.generate(path)
        costs = compute_reference_costs(domain, path, ref_search_list, heuristic,
            max_problem_time)
        if len(costs) > 0:
            cost = min(costs)
        else:
            cost = -1
        return (path, cost)
    else:
        if get_problem.problem_set is None:
            get_problem.problem_set = glob.glob(problem_dir + '/p*.pddl')
            get_problem.costs = np.load(problem_dir + '/costs.npy').item()
        problem = random.choice(get_problem.problem_set)
        print(problem)
        return (problem, get_problem.costs[problem.split('/')[-1]])


get_problem.problem_set = None
get_problem.costs = None


def compute_sqr_ipc_reward(plan_cost, reference_cost):
    ipc_reward = compute_ipc_reward(plan_cost, reference_cost)
    squared = ipc_reward * ipc_reward
    if squared >= 2.0:
        return 2.0
    return squared

def compute_time_reward(problem_time):
    if problem_time < 0:
        return 0.0
    else:
        return max_problem_time - problem_time



def gradient_update(params, action_count, reward):
    
    update = np.zeros(params.shape)
    for state_params, actions, update_row, state_reward in zip(params, action_count, update, reward):
        pi = softmax(state_params)
        state_params0 = np.copy(state_params)
        for i in range(N_ACTIONS):
            gradient = np.array([0.0] * N_ACTIONS)
            for j in range(N_ACTIONS):
                if i == j:
                    gradient[j] = pi[j] * (1 - pi[i])
                else:
                    gradient[j] = pi[j] * (- pi[i])
            update_row += learning_rate * state_reward * actions[i] * gradient / pi[i]
    return update

def replay_update(params, history, avg_reward):
    update = np.zeros(params.shape)
    for s in range(N_SAMPLES):
        (action_count, reward) = random.choice(history) #history[-1]
        centered_reward = reward - avg_reward
        update += gradient_update(params, action_count, centered_reward)
    update /= N_SAMPLES
    return update




log = open('rl_driver_log.txt', 'w')
debug_log = open('rl_debug_log.txt', 'w')
reward_log = open('rl_reward_log.txt', 'w')

params = np.zeros((n_states(), N_ACTIONS))
weight_path = Path('weights.txt')
if weight_path.exists():
    weight_file = open('weights.txt')
    weights = [float(x) for x in weight_file.read().split()]
    weight_file.close()
    params = np.array(weights).reshape(n_states(), N_ACTIONS)
else:
    save_weights(params)

n_iter = 0
avg_rewards = [(0,0.0)] * n_states() 
history = []
total_action_count = np.zeros(STATE_SPACE+(N_ACTIONS,))

search_time = int(1000 * max_problem_time - preprocessing_time)
print('Search time:', search_time)

start_time = time.time()

while time.time() - start_time < training_time:
    
    problem_path, reference_cost = get_problem()
    update = np.zeros(params.shape)
    
    for i in range(RUNS_PER_PROBLEM):
        problem_time = 0.0
        plan_cost = -1
        try:
            problem_start = time.time()
            planner_output = get_output_with_timeout(['../fast-downward.py',
                '--build', 'release64',  domain,  problem_path,
                '--heuristic', heuristic, '--search',  search % search_time],
                max_problem_time)
            problem_time = time.time() - problem_start
            print('Problem time: ', problem_time)
            plan_cost = get_cost(planner_output)
        except subprocess.TimeoutExpired:
            print('Failed to find a solution.')
            problem_time = -1
        
        if plan_cost == 0:
            continue
        
        print ('Plan cost:', plan_cost)
        print ('Reference cost:', reference_cost)
        try:
            reward = compute_ipc_reward(plan_cost, reference_cost)
            #reward = compute_sqr_ipc_reward(plan_cost, reference_cost)
            #reward = compute_time_reward(problem_time)
        except NoRewardException:
            continue
        
        reward_log.write(str(reward))
        reward_log.write('\n')
        reward_log.flush()
        
        action_count = np.zeros(STATE_SPACE+(N_ACTIONS,))
        trace_file = open('trace.txt')
        lines = trace_file.readlines()
        for l in lines:
            numbers = [ int(x) for x in l.split()]
            action_count[tuple(numbers)] += 1
        debug_log.write(str(action_count.sum()))
        debug_log.write(' ')
        total_action_count += action_count
        action_sum = action_count.sum()
        if action_sum < 0.5:
            print('The trace is empty.')
            debug_log.write('The trace is empty.\n')
            continue
        
        flat_action_count = action_count.copy().reshape(-1, N_ACTIONS)
        for i, row in enumerate(flat_action_count):
            row_sum = row.sum()
            if row_sum > 0.0:
                # normalize the row
                # row /= row_sum
                # update the baseline for the state
                (old_iter, old_reward) = avg_rewards[i]
                new_iter = old_iter + 1
                new_reward = old_reward * (old_iter / new_iter) + reward * (1 / new_iter)
                avg_rewards[i] = (new_iter, new_reward)
        # c_reward is an array now
        c_reward = reward - np.array([x for (_, x) in avg_rewards])
        print('Reward: ', reward, 'Centered: ', c_reward)
        print('Action counts:')
        print(action_count)
        #history.append((flat_action_count, reward))
        
        #update = gradient_update(params, history, avg_reward)
        partial_update = gradient_update(params, flat_action_count, c_reward)
        print('Partial update:')
        print(partial_update)
        update += partial_update
        
    
    print('Complete update:')
    print(update)
    params += update

    print('Weights:')
    print(params)
    save_weights(params)
    
    debug_log.write(str(np.isnan(params).any()))
    debug_log.write('\n')
    debug_log.flush()
    
    for row in params:
        log.write(' '.join([str(x) for x in row.tolist()]))
        log.write(' ')
    log.write('\n')
    log.flush()
    
    n_iter += 1

debug_log.write(str(total_action_count))
debug_log.write('\n')
debug_log.write(str(np.sum(total_action_count,-1)))

log.close()
debug_log.close()
reward_log.close()

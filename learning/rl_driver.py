#!/usr/bin/python3

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

sys.path.append('problem-generators')
from transport_generator import TransportGenerator
from parking_generator import ParkingGenerator


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

learning_rate = 1.0
target_problem_time = 0.3
preprocessing_time = 800 # Transport(4,9)
#preprocessing_time = 2200 # Parking(9,16)

STATE_SPACE = (2,2)
N_ACTIONS = 6

N_TRUCKS = 4
N_PACKAGES = 9

N_CURBS = 9
N_CARS = 16

N_SAMPLES = 10
RUNS_PER_PROBLEM = 20

generator = TransportGenerator(N_TRUCKS, N_PACKAGES)
#generator = ParkingGenerator(N_CURBS, N_CARS)


if len(sys.argv) == 4:
    domain = sys.argv[1]
    problem = sys.argv[2]
    training_time = int(sys.argv[3])
    generate = False
else:
    training_time = int(sys.argv[1])
    problem = 'problem.pddl'
    domain = '../../../IPC/own-transport/domain.pddl'
    #domain = '../../../IPC/own-parking/domain.pddl'
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

def get_cost(planner_output):
    planner_output = planner_output.decode('utf-8')
    m = re.search('Plan cost: [0-9]+', planner_output)
    m2 = re.search('[0-9]+', m.group(0))
    cost = int(m2.group(0))
    return cost

def get_output_with_timeout(command):

    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    try:
        output = proc.communicate(timeout=10*target_problem_time)[0]
        return output
    except subprocess.TimeoutExpired:
        try:
            parent = psutil.Process(proc.pid)
            children = parent.children(recursive=True)
            for c in children:
                try:
                    c.send_signal(signal.SIGINT)
                except psutil.NoSuchProcess:
                    pass
            proc.kill()
        except psutil.NoSuchProcess:
            pass
        raise

def get_problem():
    if generate:
        generator.generate()
    return 'problem.pddl'

def n_states():
    r = 1
    for s in STATE_SPACE:
        r *= s
    return r

def compute_reference_costs(problem_path):
    costs = []
    for ref_search in ref_search_list:
        try:
            reference_output = get_output_with_timeout(['../fast-downward.py',
                '--build', 'release64',  domain,  problem_path,
                '--heuristic', heuristic, '--search',  ref_search])
            cost = get_cost(reference_output)
            print('Reference cost:', cost)
            costs.append(cost)
        except subprocess.TimeoutExpired:
            print('Failed to find a reference solution.')
    return costs

class NoRewardException(Exception):
    pass

def compute_ipc_reward(plan_cost, reference_costs):
    # If no reference solution
    if len(reference_costs) == 0:
        if plan_cost > 0:
            return 2.0
        raise NoRewardException()
    # Otherwise
    if plan_cost < 0:
        return 0.0
    elif plan_cost == 0:
        raise NoRewardException()
    else:
        return min(reference_costs) / plan_cost




def gradient_update(params, action_count, reward):
    
    update = np.zeros(params.shape)
    for state_params, actions, update_row in zip(params, action_count, update):
        pi = softmax(state_params)
        state_params0 = np.copy(state_params)
        for i in range(N_ACTIONS):
            gradient = np.array([0.0] * N_ACTIONS)
            for j in range(N_ACTIONS):
                if i == j:
                    gradient[j] = pi[i] * (1 - pi[j])
                else:
                    gradient[j] = pi[i] * (- pi[j])
            update_row += learning_rate * reward * actions[i] * gradient
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
avg_reward = 0.0
avg_problem_reward = 0.0
history = []

search_time = int(1000 * 10 * target_problem_time - preprocessing_time)
print('Search time:', search_time)

start_time = time.time()

while time.time() - start_time < training_time:
    
    problem_path = get_problem()    
    reference_costs = compute_reference_costs(problem_path)
    update = np.zeros(params.shape)
    
    if n_iter > 0:
        avg_reward += (1 / n_iter) * (avg_problem_reward - avg_reward)
    
    avg_problem_reward = 0.0
    for i in range(RUNS_PER_PROBLEM):
        problem_time = 0.0
        plan_cost = -1
        try:
            problem_start = time.time()
            planner_output = get_output_with_timeout(['../fast-downward.py',
                '--build', 'release64',  domain,  problem_path,
                '--heuristic', heuristic, '--search',  search % search_time])
            problem_time = time.time() - problem_start
            print('Problem time: ', problem_time)
            #reward = 10*target_problem_time - problem_time
            plan_cost = get_cost(planner_output)
        except subprocess.TimeoutExpired:
            print('Failed to find a solution.')
        
        if plan_cost == 0:
            continue
        
        print ('Plan cost:', plan_cost)
        try:
            reward = compute_ipc_reward(plan_cost, reference_costs)
        except NoRewardException:
            continue
        
        reward_log.write(str(reward))
        reward_log.write('\n')
        reward_log.flush()
        avg_problem_reward += reward
        
        action_count = np.zeros(STATE_SPACE+(N_ACTIONS,))
        trace_file = open('trace.txt')
        lines = trace_file.readlines()
        for l in lines:
            numbers = [ int(x) for x in l.split()]
            action_count[tuple(numbers)] += 1
        debug_log.write(str(action_count.sum()))
        debug_log.write(' ')
        action_sum = action_count.sum()
        if action_sum < 0.5:
            print('The trace is empty.')
            debug_log.write('The trace is empty.\n')
            continue
        action_count = action_count / action_sum
        
        flat_action_count = action_count.reshape(-1, N_ACTIONS)
        c_reward = reward - avg_reward
        print('Reward: ', reward, 'Centered: ', c_reward)
        print('Action counts:', action_count)
        #history.append((flat_action_count, reward))
        
        #update = gradient_update(params, history, avg_reward)
        partial_update = gradient_update(params, flat_action_count, c_reward)
        print('Partial update:')
        print(partial_update)
        update += partial_update
        
    
    print('Complete update:')
    print(update)
    if n_iter > 0: # skip the first iteration (for the lack of baseline)
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
    
    avg_problem_reward /= RUNS_PER_PROBLEM
    n_iter += 1


log.close()
debug_log.close()
reward_log.close()

#!/usr/bin/python3

import os
import psutil
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
search = 'learning(h1)'
reference_search = 'eager_greedy(h1)'

learning_rate = 1.0
target_problem_time = 2.0

N_ACTIONS = 6

N_TRUCKS = 4
N_PACKAGES = 9

N_CURBS = 10
N_CARS = 18

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
    weights_file.write(' '.join([str(x) for x in weights.tolist()]))
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
        except psutil.NoSuchProcess:
            pass
        children = parent.children(recursive=True)
        for c in children:
            c.send_signal(signal.SIGINT)
        proc.kill()
        raise

def get_problem():
    if generate:
        generator.generate()
    return 'problem.pddl'


def compute_reward(problem_time, plan_cost):

    # Compute the reference solution
    if generate or not hasattr(compute_reward, 'ref_cost'):
        compute_reward.ref_cost = -1
        try:
            reference_output = get_output_with_timeout(['../fast-downward.py',
                '--build', 'release64',  domain,  problem_path,
                '--heuristic', heuristic, '--search',  reference_search])
            compute_reward.ref_cost = get_cost(reference_output)
        except subprocess.TimeoutExpired:
            print('Failed to find a reference solution.')
            if not generate:
                sys.exit('Terminating.')
    
    # If no reference solution
    if compute_reward.ref_cost < 0:
        if plan_cost < 0:
            return 0.0
        return 2.0 # fixed reward for solving a problem without reference solution
    # Otherwise
    if plan_cost < 0:
        return 0.0
    return compute_reward.ref_cost / plan_cost



log = open('rl_driver_log.txt', 'w')

start_time = time.time()

params = np.array([0.0] * N_ACTIONS)
weight_path = Path('weights.txt')
if weight_path.exists():
    weight_file = open('weights.txt')
    weights = [float(x) for x in weight_file.read().split()]
    weight_file.close()
    for i in range(len(weights)):
        params[i] = weights[i]
else:
    save_weights(params)

returns = []

while time.time() - start_time < training_time:
    
    problem_path = get_problem()    
    
    problem_time = 0.0
    plan_cost = -1
    try:
        problem_start = time.time()
        planner_output = get_output_with_timeout(['../fast-downward.py',
            '--build', 'release64',  domain,  problem_path,
            '--heuristic', heuristic, '--search',  search])
        problem_time = time.time() - problem_start
        print('Problem time: ', problem_time)
        #reward = 10*target_problem_time - problem_time
        plan_cost = get_cost(planner_output)
    except subprocess.TimeoutExpired:
        print('Failed to find a solution.')
    
    reward = compute_reward(problem_time, plan_cost)
    returns.append(reward)
    if len(returns) > 100:
        returns = returns[-100:]
    
    action_count = [0] * N_ACTIONS
    trace_file = open('trace.txt')
    lines = trace_file.readlines()
    for l in lines:
        action_count[int(l.split()[0])] += 1
    action_count = np.array(action_count)
    action_count = action_count / action_count.sum()
    
    # Policy gradient
    pi = softmax(params)
    params0 = np.copy(params)
    centered_reward = reward - np.mean(np.array(returns))
    for i in range(N_ACTIONS):
        gradient = np.array([0.0] * N_ACTIONS)
        for j in range(N_ACTIONS):
            if i == j:
                gradient[j] = pi[i] * (1 - pi[j])
            else:
                gradient[j] = pi[i] * (- pi[j])
        params += learning_rate * centered_reward * action_count[i] * gradient

    print('Reward: ', reward, 'Centered: ', centered_reward)
    print('Action counts:', action_count)
    print('Update:', params - params0)
    print('Weights:', params)
    save_weights(params)
    
    log.write(' '.join([str(x) for x in params.tolist()]))
    log.write('\n')
    log.flush()
    
    np.append(returns, reward)


log.close()

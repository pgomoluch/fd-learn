#!/usr/bin/python3

import os
import subprocess
import sys
import time

import numpy as np

sys.path.append('problem-generators')
from transport_generator import TransportGenerator
from parking_generator import ParkingGenerator


heuristic = 'h1=ff(transform=adapt_costs(one))'
search = 'learning(h1)'

learning_rate = 1.0
target_problem_time = 2.0

N_ACTIONS = 6

N_TRUCKS = 4
N_PACKAGES = 9

N_CURBS = 10
N_CARS = 18

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

log = open('rl_driver_log.txt', 'w')

generator = TransportGenerator(N_TRUCKS, N_PACKAGES)
#generator = ParkingGenerator(N_CURBS, N_CARS)

start_time = time.time()

params = np.array([0.0] * N_ACTIONS)
save_weights(params)

returns = []

while time.time() - start_time < training_time:
    
    if generate:
        generator.generate()
    problem_start = time.time()
    
    reward = 0.0
    problem_time = 0.0
    try:
        subprocess.call(['nohup', '../fast-downward.py', '--build', 'release64',  domain,  problem,
            '--heuristic', heuristic, '--search',  search],
            timeout=10*target_problem_time)
        problem_time = time.time() - problem_start
        reward = 10*target_problem_time - problem_time
        if reward < 0.0:
            reward = 0.0
    except subprocess.TimeoutExpired:
        #generator.easier()
        continue
    
    returns.append(reward)
    if len(returns) > 100:
        returns = returns[-100:]
    
    problem_time = time.time() - problem_start
    episode_file = open('episode.txt')
    action_count = episode_file.readline()
    episode_file.close()
    action_count = np.array([int(x) for x in action_count.split()])
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

    print('Return:', centered_reward, 'Action counts:', action_count)
    print('Update:', params - params0)
    print('Weights:', params)
    save_weights(params)
    
    log.write(' '.join([str(x) for x in params.tolist()]))
    log.write('\n')
    log.flush()
    
    np.append(returns, reward)
    
    if generate:
        if problem_time > target_problem_time:
            pass
            #generator.easier()
        else:
            pass
            #generator.harder()

log.close()

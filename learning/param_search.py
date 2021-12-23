#!/usr/bin/python3

import ast
import errno
import glob
import math
import numpy as np
import os
import random
import subprocess
import sys
import time
from configparser import ConfigParser

from rl_common import get_output_with_timeout, get_cost, compute_reference_costs, compute_ipc_reward, save_params
from evaluators.condor_evaluator import CondorEvaluator
from evaluators.sequential_evaluator import SequentialEvaluator
from evaluators.parallel_evaluator import ParallelEvaluator
from evaluators.mpi_evaluator import MPIEvaluator

from problem_generators.transport_generator import TransportGenerator
from problem_generators.parking_generator import ParkingGenerator
from problem_generators.elevators_generator import ElevatorsGenerator
from problem_generators.nomystery_generator import NomysteryGenerator
from problem_generators.floortile_generator import FloortileGenerator

from parameter_handlers.direct_parameter_handler import DirectParameterHandler
from parameter_handlers.neural_parameter_handler import NeuralParameterHandler

HEURISTIC = 'h1=ff(transform=adapt_costs(one))'
SEARCH = 'parametrized(h1,params=%s,scales={scales_path})'
SEARCH = SEARCH.format(
    scales_path = os.path.abspath(os.path.join(os.getcwd(), 'scales.txt'))
)

#param_handler = DirectParameterHandler()
param_handler = NeuralParameterHandler()

# Random walk optimization
INITIAL_PARAMS = [0.2, 5, 0.5, 20]
UNITS = [0.05, 1, 0.05, 5]
MIN_PARAMS = [0.0, 0.0, 0.0, 0.0]
MAX_PARAMS = [1.0, float('inf'), 1.0, float('inf')]

POPULATION_SIZE = 50
ELITE_SIZE = 10
N_TEST_PROBLEMS = 20
RUNS_PER_PROBLEM = 4
MAX_PROBLEM_TIME = 180.0
ALPHA = 0.7

OUTPUT_PATH_PREFIX = ''#os.environ['PBS_O_WORKDIR'] # leave empty to write output files in the working dir
STATE_FILE_PATH = os.path.join(OUTPUT_PATH_PREFIX, 'search_state.npz')
ALL_PROBLEMS = True
GENERATE_PROBLEMS = True
PARAMS_DIR = os.path.join(OUTPUT_PATH_PREFIX, 'params')

if GENERATE_PROBLEMS:
    generator_class = TransportGenerator
    generator_key = 'agr2019'
    # generator = TransportGenerator(4, 11, 30)
    # difficulty_level = -7
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

def get_all_problems():
    problems = glob.glob(PROBLEM_DIR + '/p*.pddl')
    problems.sort()
    return [ (p, -1) for p in problems ]

def cem_evolution_step(mean, cov, params, sorted_ids):
    best_ids = sorted_ids[:ELITE_SIZE]
    new_mean = (1-ALPHA) * mean + ALPHA * np.mean(params[best_ids], 0)
    new_cov = (1-ALPHA) * cov + ALPHA * np.cov(params[best_ids], rowvar=False)
    return (new_mean, new_cov)

def fixed_variance_evolution_step(mean, cov, params, sorted_ids):
    best_ids = sorted_ids[:ELITE_SIZE]
    new_mean = (1-ALPHA) * mean + ALPHA * np.mean(params[best_ids], 0)
    return (new_mean, cov)

def canonical_evolution_step(mean, cov, params, sorted_ids):
    
    if not canonical_evolution_step.weights:
        w = []
        m = len(sorted_ids)
        for i in range(m):
            w.append(math.log(m+0.5) - math.log(i+1))
        s = sum(w)
        w = [x/s for x in w]
        canonical_evolution_step.weights = w
        print('CES weights:', canonical_evolution_step.weights)
    
    new_mean = np.zeros(mean.shape)
    for i in sorted_ids:
        new_mean += canonical_evolution_step.weights[i] * params[i]
    
    return (new_mean, cov)

canonical_evolution_step.weights = None



conf = ConfigParser()
conf.read(sys.argv[1])

DOMAIN = conf['plan']['domain']
PROBLEM_DIR = conf['plan']['problem_dir']
TRAINING_TIME = conf['opt'].getfloat('training_time')
state_file_entry = conf['opt'].get('state_file', None)

continuing = False
if state_file_entry:
    continuing = True
    STATE_FILE_PATH = state_file_entry


start_time = time.time()
params_log = open(os.path.join(OUTPUT_PATH_PREFIX, 'params_log.txt'), 'w')
condor_log = open(os.path.join(OUTPUT_PATH_PREFIX, 'condor_log.txt'), 'w')

evaluator = MPIEvaluator(
    population_size = POPULATION_SIZE,
    n_test_problems = N_TEST_PROBLEMS,
    domain_path = DOMAIN,
    heuristic_str = HEURISTIC,
    search_str = SEARCH,
    max_problem_time = MAX_PROBLEM_TIME,
    param_handler = param_handler)

if continuing:
    state_file = open(STATE_FILE_PATH, 'rb')
    npz_content = np.load(state_file)
    mean = npz_content['mean']
    cov = npz_content['cov']
    state_file.close()
else:
    mean = np.array(param_handler.initial_mean)
    cov = np.diag(np.square(param_handler.initial_stddev))

np.set_printoptions(suppress=True,precision=4)

#os.makedirs(PARAMS_DIR, exist_ok=True) # python3
try:
    os.makedirs(PARAMS_DIR)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

kinit_time = 0
iteration = 0
while time.time() - start_time < TRAINING_TIME:
    
    # Renew Kerberos ticket
    #if time.time() - kinit_time > 5 * 3600:
    #    subprocess.Popen("cat pass.txt | kinit", shell=True)
    #    kinit_time = time.time()
    
    print('Mean: ', mean)
    print('Covariance:\n', cov)
    
    # Choose the test problems
    if ALL_PROBLEMS:
        if GENERATE_PROBLEMS:
            generator_class.generate_series(generator_key, PROBLEM_DIR)
        paths_and_costs = get_all_problems()
    else:
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
        param_handler.bound_params(row)
    # For debugging: fix the parameters
    #params = np.concatenate((
    #    np.tile(np.array([[0.2,0,1000,0.0]]), (POPULATION_SIZE//2, 1)),
    #    np.tile(np.array([[0.2,20,100,0.9]]), (POPULATION_SIZE//2, 1))),
    #    axis=0)

    scores = evaluator.score_params(params, paths_and_costs, condor_log)
    
    scores = np.array(scores)
    sorted_ids = np.argsort(-scores)
    
    # Only update the distribution if some problems have been solved
    if scores[sorted_ids[0]] > 0.001:
        mean, cov = cem_evolution_step(mean, cov, params, sorted_ids)

    condor_log.write('Best parameters:\n')
    for i in sorted_ids:
        condor_log.write(str(params[i]) + ' ' + str(scores[i]) + ' ' + str(i) + '\n')
    condor_log.write('\n')
    condor_log.flush()
    params_log.write(str(mean) + '\n' + str(cov) + '\n\n')
    params_log.flush()
    
    param_handler.save_params(mean, 'params.txt')
    if PARAMS_DIR:
        param_handler.save_params(mean, os.path.join(PARAMS_DIR, 'params%d.txt' % iteration))
    state_file = open(STATE_FILE_PATH, 'wb')
    np.savez(state_file, mean=mean, cov=cov)
    state_file.close()
    
    if GENERATE_PROBLEMS and not ALL_PROBLEMS:
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
    
    iteration += 1
    
params_log.close()
condor_log.close()


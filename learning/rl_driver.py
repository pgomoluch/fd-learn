#!/usr/bin/python3

import subprocess
import sys
import time

conf = 'learning(ff,learning_rate=%f)'
learning_rate = 0.1
transport_generator = '../../../IPC/own-transport/generator14L/city-generator.py'

if len(sys.argv) == 4:
    domain = sys.argv[1]
    problem = sys.argv[2]
    iterations = int(sys.argv[3])
    generate = False
else:
    iterations = int(sys.argv[1])
    problem = 'problem.pddl'
    domain = '../../../IPC/own-transport/domain.pddl'
    generate = True


def generate_transport_problem():
    ipc_generator_command = ['python', transport_generator, '15', '1000', '3', '100', '3', '6', str(time.time())]
    problem = subprocess.check_output(ipc_generator_command).decode('utf-8')
    problem_file = open('problem.pddl', 'w')
    problem_file.write(problem)
    problem_file.close()


log = open('rl_driver_log.txt', 'w')

for i in range(1,iterations+1):
    print(i)
    if generate:
        generate_transport_problem()
    subprocess.call(['../fast-downward.py', '--build', 'release64',  domain,  problem, '--search',  conf % learning_rate])
    weights = open('weights.txt')
    log.write(weights.read())
    log.write('\n')
    log.flush()
    weights.close()
    if i % 100 == 0:
        learning_rate /= 2

log.close()


#!/usr/bin/python3

import subprocess
import sys
import time

conf = 'learning(ff,learning_rate=%f)'
#conf = 'learning(ff(transform=adapt_costs(one)),learning_rate=%f)'
learning_rate = 0.001
transport_generator = '../../../IPC/own-transport/generator14L/city-generator.py'
target_problem_time = 2.0


if len(sys.argv) == 4:
    domain = sys.argv[1]
    problem = sys.argv[2]
    training_time = int(sys.argv[3])
    generate = False
else:
    training_time = int(sys.argv[1])
    problem = 'problem.pddl'
    domain = '../../../IPC/own-transport/domain.pddl'
    generate = True


class TransportGenerator:
    
    def __init__(self):
        self.trucks = 3
        self.packages = 6
    
    def generate(self):
        ipc_generator_command = ['python', transport_generator, '15', '1000', '3', '100', str(self.trucks), str(self.packages), str(time.time())]
        problem = subprocess.check_output(ipc_generator_command).decode('utf-8')
        problem_file = open('problem.pddl', 'w')
        problem_file.write(problem)
        problem_file.close()
    
    def easier(self):
        if self.packages > 1:
            self.packages -= 1
    
    def harder(self):
        self.packages += 1


log = open('rl_driver_log.txt', 'w')
problem_log = open('rl_driver_problem_log.txt', 'w')

generator = TransportGenerator()

start_time = time.time()

while time.time() - start_time < training_time:
    if generate:
        generator.generate()
    problem_start = time.time()
    
    try:
        subprocess.call(['../fast-downward.py', '--build', 'release64',  domain,  problem, '--search',  conf % learning_rate], timeout=10*target_problem_time)
    except subprocess.TimeoutExpired:
        generator.easier()
        continue
    
    problem_time = time.time() - problem_start
    problem_log.write('%f (%d)\n' % (problem_time, generator.packages))
    problem_log.flush()
    weights = open('weights.txt')
    log.write(weights.read())
    log.write('\n')
    log.flush()
    weights.close()
    if generate:
        if problem_time > target_problem_time:
            generator.easier()
        else:
            generator.harder()
    #if i % 100 == 0:
    #    learning_rate /= 2

log.close()
problem_log.close()

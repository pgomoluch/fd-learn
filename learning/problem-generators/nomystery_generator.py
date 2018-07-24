import os
import time
import subprocess

ipc_generator = '../../../IPC/own-no-mystery/generator/nomystery -l %d -p %d -c %d -s %d'

class NomysteryGenerator:

    def __init__(self, locations, packages, constraint):
        self.locations = locations
        self.packages = packages
        self.constraint = constraint
        self.counter = 0
    
    def generate(self, result_path = 'problem.pddl'): #, seed = None
        seed = time.time() + self.counter * 1000
        comm = ipc_generator % (self.locations, self.packages, self.constraint, seed)
        problem = subprocess.check_output(comm.split(' ')).decode('utf-8')
        problem_file = open(result_path, 'w')
        problem_file.write(problem)
        problem_file.close()
        self.counter += 1
    
    def generate_batch(self, n, base_path = 'problem'):
        for i in range(1, n+1):
            path = base_path + str(i) + '.pddl'
            self.generate(path)

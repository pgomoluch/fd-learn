import os
import time
import subprocess

from .base_generator import BaseGenerator

ipc_generator = '../../../IPC/own-no-mystery/generator/nomystery -l %d -p %d -c %f -s %d'

class NomysteryGenerator(BaseGenerator):
    
    counter = 0
    
    def __init__(self, locations, packages, constraint):
        self.locations = locations
        self.packages = packages
        self.constraint = constraint
    
    def __str__(self):
        return 'NomysteryGenerator(%d locations, %d packages, %f constraint)' % (
            self.packages, self.locations, self.constraint)
    
    def generate(self, result_path = 'problem.pddl'): #, seed = None
        seed = time.time() + NomysteryGenerator.counter * 1000
        comm = ipc_generator % (self.locations, self.packages, self.constraint, seed)
        problem = subprocess.check_output(comm.split(' ')).decode('utf-8')
        problem_file = open(result_path, 'w')
        problem_file.write(problem)
        problem_file.close()
        NomysteryGenerator.counter += 1
    
    def easier(self):
        if self.packages > 1:
            self.packages -= 1
    
    def harder(self):
        self.packages += 1
    
    conf = {
        'ipc2011': ((6,6,1.5), (7,7,1.5), (8,8,1.5), (9,9,1.5), (10,10,1.5),
            (11,11,1.5), (12,12,1.5), (13,13,1.5), (14,14,1.5), (15,15,1.5),
            (6,6,1.1), (7,7,1.1), (8,8,1.1), (9,9,1.1), (10,10,1.1),
            (11,11,1.1), (12,12,1.1), (13,13,1.1), (14,14,1.1), (15,15,1.1)),
        'agr2019': ((6,8,1.0),) * 10
    }

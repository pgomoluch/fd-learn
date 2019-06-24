import os
import time
import subprocess

from .base_generator import BaseGenerator

ipc_generator = '../../../IPC/own-floortile/floortile-generator.py'

class FloortileGenerator(BaseGenerator):

    def __init__(self, rows, columns, robots):
        self.rows = rows
        self.columns = columns
        self.robots = robots
    
    def __str__(self):
        return 'FloortileGenerator(%d rows, %d columns, %d robots)' % (
            self.rows, self.columns, self.robots)
    
    def generate(self, result_path = 'problem.pddl'):
        ipc_generator_command = ['python', ipc_generator, 'problem',
           str(self.rows), str(self.columns), str(self.robots), 'seq']
        problem = subprocess.check_output(ipc_generator_command).decode('utf-8')
        problem_file = open(result_path, 'w')
        problem_file.write(problem)
        problem_file.close()
    
    def easier(self):
        if self.rows > self.columns + 1:
            self.rows -= 1
        else:
            self.columns -= 1
    
    def harder(self):
        if self.rows > self.columns + 1:
            self.columns += 1
        else:
            self.rows += 1
    
    conf = {
        'ipc2011': ((5,3,2), (5,3,2), (6,3,2), (6,3,2), (5,4,2), (5,4,2),
            (6,4,2), (6,4,2), (7,4,2), (7,4,2), (6,5,2), (6,5,2),
            (7,5,3), (7,5,3), (7,6,3), (7,6,3), (8,6,3), (8,6,3),
            (8,7,3), (8,7,3)),
        'agr2019': ((4,3,2),) * 10
    }

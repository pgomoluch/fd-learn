import os
import time
import subprocess

ipc_generator = '../../../IPC/own-floortile/floortile-generator.py'

class FloortileGenerator:

    def __init__(self, rows, columns, robots):
        self.rows = rows
        self.columns = columns
        self.robots = robots
    
    def generate(self, result_path = 'problem.pddl'):
        ipc_generator_command = ['python', ipc_generator, 'problem',
           str(self.rows), str(self.columns), str(self.robots), 'seq']
        problem = subprocess.check_output(ipc_generator_command).decode('utf-8')
        problem_file = open(result_path, 'w')
        problem_file.write(problem)
        problem_file.close()
    
    def generate_batch(self, n, base_path = 'problem'):
        for i in range(1, n+1):
            path = base_path + str(i) + '.pddl'
            self.generate(path)

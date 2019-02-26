import os
import time
import subprocess

ipc_generator = '../../../IPC/own-parking/parking-generator.pl'

class ParkingGenerator:
    
    def __init__(self, curbs, cars):
        self.curbs = curbs
        self.cars = cars
    
    def __str__(self):
        return 'ParkingGenerator(%d curbs, %d cars)' % (self.curbs, self.cars)
        
    def generate(self, result_path = 'problem.pddl'):
        ipc_generator_command = ['perl', ipc_generator,
            'p', str(self.curbs), str(self.cars)]
        problem = subprocess.check_output(ipc_generator_command).decode('utf-8')
        problem_file = open(result_path, 'w')
        problem_file.write(problem)
        problem_file.close()
    
    def generate_batch(self, n, base_path = 'problem'):
        for i in range(1, n+1):
            path = base_path + str(i) + '.pddl'
            self.generate(path)
    
    def easier(self):
        if self.curbs > 3:
            self.curbs -= 1
            self.cars -= 2
    
    def harder(self):
        self.curbs += 1
        self.cars += 2

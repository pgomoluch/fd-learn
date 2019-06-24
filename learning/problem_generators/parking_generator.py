import os
import time
import subprocess

from .base_generator import BaseGenerator

ipc_generator = '../../../IPC/own-parking/parking-generator.pl'

class ParkingGenerator(BaseGenerator):
    
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
    
    def easier(self):
        if self.curbs > 3:
            self.curbs -= 1
            self.cars -= 2
    
    def harder(self):
        self.curbs += 1
        self.cars += 2
    
    conf = {
        'ipc2014': ((15,28), (15,28), (16,30), (16,30), (16,30),
            (17,32), (17,32), (17,32), (18,34), (18,34), (18,34),
            (19,36), (19,36), (19,36), (20,38), (20,38), (20,38),
            (21,40), (21,40), (21,40)),
        'agr2019': ((10,18),) * 10
    }

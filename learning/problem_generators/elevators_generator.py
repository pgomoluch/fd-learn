import os
import time
import subprocess

ipc_generator1 = '../../../IPC/own-elevators/generator/generate_elevators {passengers} {passengers} 1 1 1 {floors} {area_size} {n_fast} {n_slow} {fast_capacity} {slow_capacity}'
ipc_generator2 = '../../../IPC/own-elevators/generator/generate_pddl {floors} {floors} 1 {passengers} {passengers} 1 1 1'

class ElevatorsGenerator:

    SLOW_CAPACITY = 4
    FAST_CAPACITY = 4

    def __init__(self, passengers, floors, area_size, n_fast, n_slow):
        self.params = {'passengers': passengers, 'floors': floors,
            'area_size': area_size, 'n_fast': n_fast, 'n_slow': n_slow,
            'slow_capacity': self.SLOW_CAPACITY,
            'fast_capacity': self.FAST_CAPACITY}
    
    def __str__(self):
        return 'ElevatorsGenerator' + str(self.params)

    def generate(self, result_path = 'problem.pddl'):
        comm1 = ipc_generator1.format(**self.params).split(' ')
        comm2 = ipc_generator2.format(**self.params).split(' ')
        subprocess.call(comm1, stdout = subprocess.DEVNULL)
        subprocess.call(comm2, stdout = subprocess.DEVNULL)
        base_name = 'p{floors}_{passengers}_1'.format(**self.params)
        os.rename(base_name + '.pddl', result_path)
        os.remove(base_name + '.txt')

    def generate_batch(self, n, base_path = 'problem'):
        for i in range(1, n+1):
            path = base_path + str(i) + '.pddl'
            self.generate(path)
            time.sleep(1.1)
    
    def easier(self):
        if self.params['passengers'] > 1:
            self.params['passengers'] -= 1

    def harder(self):
        self.params['passengers'] += 1

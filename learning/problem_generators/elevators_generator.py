import os
import time
import subprocess

from .base_generator import BaseGenerator

ipc_generator1 = '../../../IPC/own-elevators/generator/generate_elevators {passengers} {passengers} 1 1 1 {floors} {area_size} {n_fast} {n_slow} {fast_capacity} {slow_capacity}'
ipc_generator2 = '../../../IPC/own-elevators/generator/generate_pddl {floors} {floors} 1 {passengers} {passengers} 1 1 1'

class ElevatorsGenerator(BaseGenerator):

    def __init__(self, passengers, floors, area_size, n_fast, n_slow, fast_cap=6, slow_cap=4):
        self.params = {'passengers': passengers, 'floors': floors,
            'area_size': area_size, 'n_fast': n_fast, 'n_slow': n_slow,
            'slow_capacity': slow_cap,
            'fast_capacity': fast_cap}
    
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
            self.params['passengers'] -= 5

    def harder(self):
        self.params['passengers'] += 5
    
    conf = {
        'ipc2011': ((14,16,8,2,1,4,3), (21,24,8,2,1,6,4), (27,24,8,2,1,6,4),
            (26,16,8,2,1,4,3), (22,16,8,2,1,4,3), (30,24,8,2,1,6,4),
            (24,24,8,2,1,6,4), (33,24,8,2,1,6,4), (36,24,8,2,1,6,4),
            (39,24,8,2,1,6,4), (40,24,6,4,1,6,4), (43,24,6,4,1,6,4),
            (46,24,6,4,1,6,4), (49,24,6,4,1,6,4), (52,24,6,4,1,6,4),
            (40,40,10,4,1,6,4), (45,40,10,4,1,6,4), (50,40,10,4,1,6,4),
            (55,40,10,4,1,6,4), (60,40,10,4,1,6,4)),
        'agr2019': ((27,12,6,2,2,4,4),) * 10
    }

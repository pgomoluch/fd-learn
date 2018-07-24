#!/usr/bin/python3

import subprocess
import sys
import time

sys.path.append('problem-generators')
from transport_generator import TransportGenerator
from parking_generator import ParkingGenerator
from elevators_generator import ElevatorsGenerator
from nomystery_generator import NomysteryGenerator

N_TRUCKS = 4
N_PACKAGES = 9

N_CURBS = 9
N_CARS = 16

N_SAMPLES = 10
PROBLEM_PATH = 'problem.pddl'


generator = TransportGenerator(N_TRUCKS, N_PACKAGES)
#generator = ParkingGenerator(N_CURBS, N_CARS)
#generator = ElevatorsGenerator(20,12,6,2,2)
#generator = NomysteryGenerator(6,7,1.3)

domain = '../../../IPC/own-transport/domain.pddl'
#domain = '../../../IPC/own-parking/domain.pddl'
#domain = '../../../IPC/own-elevators/domain.pddl'
#domain = '../../../IPC/own-no-mystery/domain.pddl'


iterations = 10
total_time = 0.0
for i in range(N_SAMPLES):
    generator.generate(PROBLEM_PATH)
    start = time.time()
    subprocess.check_call(['../fast-downward.py', '--build', 'release64',
        '--translate', '--preprocess', domain, PROBLEM_PATH])
    total_time += (time.time() - start)
print(int(1000 * total_time / N_SAMPLES), 'ms')

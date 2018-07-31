#!/usr/bin/python3

import bisect
import psutil
import subprocess
import sys
import signal
import time

sys.path.append('problem-generators')
from transport_generator import TransportGenerator
from parking_generator import ParkingGenerator
from elevators_generator import ElevatorsGenerator
from nomystery_generator import NomysteryGenerator

iterations = 100
heuristic = 'h1=ff(transform=adapt_costs(one))'
search = 'eager_greedy(h1)'
problem_path = 'problem.pddl'
domain_path = '../../../IPC/own-transport/domain.pddl'
#domain_path = '../../../IPC/own-parking/domain.pddl'
#domain_path = '../../../IPC/own-elevators/domain.pddl'
#domain_path = '../../../IPC/own-no-mystery/domain.pddl'
# Transport
confs = [    
    (4,7), (4,8), (4,9), (4,10),
    (5,7), (5,8), (5,9), (5,10),
    (6,7), (6,8), (6,9), (6,10)
]
# Parking
#confs = [
#    (9,10),(9,12),(9,14),(9,16),
#    (10,12),(10,14),(10,16),(10,18),
#    (11,14),(11,16),(11,18),(11,20)
#]
# Elevators
#confs = [(10,12,3,1,1), (15,12,3,1,1), (20,12,3,1,1),
#         (10,12,6,1,1), (15,12,6,1,1), (20,12,6,1,1),
#         (10,24,3,1,1), (15,24,3,1,1), (20,24,3,1,1),
#         (10,24,6,1,1), (15,24,6,1,1), (20,24,6,1,1)]
# No-mystery
#confs = [(6,7,1.3), (7,6,1.3), (7,7,1.3)]

target_times = [5.0, 60.0]


def run_with_timeout(command, timeout):
    
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    try:
        proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            parent = psutil.Process(proc.pid)
        except psutil.NoSuchProcess:
            pass
        children = parent.children(recursive=True)
        for c in children:
            c.send_signal(signal.SIGINT)
        proc.kill()
        raise

print('Duration upper bound:', len(confs)*iterations*max(target_times))
print('Conf | Avg time | Median time |', ' | '.join([str(t) for t in target_times]))
for c in confs:
    times = []
    solved = 0
    solved_in_time = 0
    
    generator = TransportGenerator(*c)
    #generator = ParkingGenerator(*c)
    #generator = ElevatorsGenerator(*c)
    #generator = NoMysteryGenerator(*c)
    
    for i in range(iterations):    
        generator.generate(problem_path)
        start_time = time.time()
        try:
            run_with_timeout(['../fast-downward.py',
                '--build', 'release64',  domain_path,  problem_path,
                '--heuristic', heuristic, '--search',  search],
                max(target_times))
            problem_time = time.time() - start_time
        except subprocess.TimeoutExpired:
            problem_time = max(target_times)
        times.append(problem_time)

    times.sort()
    solved_by_time = [ bisect.bisect_left(times, x) for x in target_times ]
    print(c, round(sum(times)/iterations, 2), ' ', round(times[int(iterations/2)],2), ' ',
        ' '.join([str(x / iterations) for x in solved_by_time]))
    sys.stdout.flush()


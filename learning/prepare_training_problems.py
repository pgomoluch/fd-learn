#!/usr/bin/python3

import numpy as np
import sys

from rl_common import get_output_with_timeout, get_cost, compute_reference_costs

sys.path.append('problem-generators')
from transport_generator import TransportGenerator
from parking_generator import ParkingGenerator
from elevators_generator import ElevatorsGenerator
from nomystery_generator import NomysteryGenerator


problem_dir = '../../training/t4t11p'
n_problems = 2
timeout = 5.0


heuristic = 'h1=ff(transform=adapt_costs(one))'
search = 'learning(h1,t=%d)'
ref_search1 = 'eager_greedy(h1)'
ref_search2 = 'learning(h1,weights=weights2of5.txt)'
ref_search3 = 'learning(h1,weights=weights3of5.txt)'
ref_search4 = 'learning(h1,weights=weights4of5.txt)'
ref_search5 = 'learning(h1,weights=weights5of5.txt)'
ref_search_list = [ref_search1, ref_search2, ref_search3,
    ref_search4, ref_search5]

generator = TransportGenerator(4, 11)
#generator = ParkingGenerator(10, 18)
#generator = ElevatorsGenerator(27,12,6,2,2)
#generator = NomysteryGenerator(6,8,1.3)

domain = '../../../IPC/own-transport/domain.pddl'
#domain = '../../../IPC/own-parking/domain.pddl'
#domain = '../../../IPC/own-elevators/domain.pddl'
#domain = '../../../IPC/own-no-mystery/domain.pddl'

basename = problem_dir + '/p'
generator.generate_batch(n_problems, basename)

all_costs = dict()
for i in range(1, n_problems+1):
    problem_path = basename + str(i) + '.pddl'
    problem_costs = compute_reference_costs(domain, problem_path, ref_search_list, heuristic, timeout)
    dict_key = 'p' + str(i) + '.pddl'
    if len(problem_costs) > 0:
        all_costs[dict_key] = min(problem_costs)
    else:
        all_costs[dict_key] = -1

np.save(problem_dir + '/costs.npy', all_costs)

# Find the feature scales to use with parametrized search, given a search config file
# Usage: python find_scales.py <config file> <output path>

import ast
import glob
import numpy as np
import os
import subprocess
import sys
from configparser import ConfigParser

from problem_generators.transport_generator import TransportGenerator
from problem_generators.parking_generator import ParkingGenerator
from problem_generators.elevators_generator import ElevatorsGenerator
from problem_generators.nomystery_generator import NomysteryGenerator
from problem_generators.floortile_generator import FloortileGenerator


generator_dict = {
    'transport': TransportGenerator,
    'parking': ParkingGenerator,
    'elevators': ElevatorsGenerator,
    'no-mystery': NomysteryGenerator,
    'floortile': FloortileGenerator
}

HEURISTIC = 'h1=ff(transform=adapt_costs(one))'
SEARCH = 'parametrized(h1,neural=False,log=log.txt)'
N_PROBLEMS = 10

conf = ConfigParser()
conf.read(sys.argv[1])
output_path = sys.argv[2]

domain_path = conf['problems']['domain_file']
problem_dir = conf['problems']['problem_dir']
max_problem_time = conf['problems'].getfloat('max_problem_time')
generator_class = conf['problems'].get('problem_generator')
generator_class = generator_dict[generator_class]
generator_key = conf['problems'].get('generator_key', None)
if not generator_key:
     generator_args = conf['problems'].get('generator_args', None)
     generator_args = ast.literal_eval(generator_args)
     generator = generator_class(*generator_args)


if generator_key:
     generator_class.generate_series(generator_key, problem_dir)
else:
     for i in range(N_PROBLEMS):
          problem_path = problem_dir + '/p' + str(i) + '.pddl'
          generator.generate(problem_path)

problems = glob.glob(problem_dir + '/p*.pddl')
max_features = []
for p in problems:
     try:
          print("Solving", p, "...")
          subprocess.check_output([
               '../fast-downward.py', '--build', 'release64',
               '--overall-time-limit', str(int(max_problem_time)),
               '--overall-memory-limit', '4G',
               domain_path, p,
               '--heuristic', HEURISTIC,
               '--search', SEARCH
          ])
          max_features.append(np.max(np.loadtxt("log.txt", dtype=int), axis=0))
     except subprocess.CalledProcessError as e:
          print('CalledProcessError:\n' + e.output.decode('utf-8'))
     

max_features = np.max(np.array(max_features), axis=0)
print("Scales:", max_features)
np.savetxt(output_path, max_features, fmt='%i')

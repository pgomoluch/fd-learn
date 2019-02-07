#!/usr/bin/python3

import os
import subprocess
import sys

from rl_common import get_cost

CONFS = [
    '--search "eager_greedy(ff(transform=adapt_costs(one)))"',
    '--search "eager_greedy(goalcount)"',
    '--search "astar(ff(transform=adapt_costs(one)))"',
    '--search "astar(goalcount)"',
    '--alias seq-sat-lama-2011'
]
TIME_LIMIT = 5
MEMORY_LIMIT = '4G'


CONDOR_DIR = 'condor_find_cost'
CONF_FILE = 'fd.sh'
CONDOR_SCRIPT_NAME = 'condor.sh'
CONDOR_SPEC_NAME = 'condor.cmd'
CONDOR_OUT = 'condor.out'
CONDOR_ERR = 'condor.err'
CONDOR_LOG = 'condor.log'

CONDOR_SCRIPT = """#!/bin/bash
echo starting
cd {condor_dir}/$1
cat {conf_file}
./{conf_file}
"""

CONDOR_SPEC = """universe = vanilla
executable = {condor_dir}/{condor_script}
output = {condor_dir}/$(Process)/{condor_out}
error = {condor_dir}/$(Process)/{condor_err}
log = {condor_dir}/{condor_log}
arguments = $(Process)
requirements = regexp("^sprite[0-9][0-9]", TARGET.Machine)
queue {n_conf}
"""

FD_PLAIN = "/usr/bin/python {fd_path} --build release64 --overall-time-limit {time_limit} --overall-memory-limit {memory_limit} {domain} {problem} {conf}"

FD_ALIAS = "/usr/bin/python {fd_path} --build release64 {conf} --overall-time-limit {time_limit} --overall-memory-limit {memory_limit} {domain} {problem}"

domain = os.path.abspath(sys.argv[1])
problem = os.path.abspath(sys.argv[2])
output_file_path = sys.argv[3]

for i, c in enumerate(CONFS):
    subdir = os.path.join(CONDOR_DIR, str(i))
    os.makedirs(subdir, exist_ok=True)
    c_path = os.path.join(subdir, CONF_FILE)
    c_file = open(c_path, 'w')
    #c_file.write(c)
    if '--alias' in c:
        c_file.write(FD_ALIAS.format(
            fd_path=os.path.abspath('../fast-downward.py'),
            conf=c,
            time_limit=TIME_LIMIT,
            memory_limit=MEMORY_LIMIT,
            domain=domain,
            problem=problem))
    else:
        c_file.write(FD_PLAIN.format(
            fd_path=os.path.abspath('../fast-downward.py'),
            time_limit=TIME_LIMIT,
            memory_limit=MEMORY_LIMIT,
            domain=domain,
            problem=problem,
            conf=c))
    c_file.close()
    os.chmod(c_path, 0o775)

condor_script_path = os.path.join(CONDOR_DIR,CONDOR_SCRIPT_NAME)
condor_script_file = open(condor_script_path, 'w')
condor_script_file.write(CONDOR_SCRIPT.format(
    condor_dir=CONDOR_DIR,
    fd_path=os.path.abspath('../fast-downward.py'),
    time_limit=TIME_LIMIT,
    memory_limit=MEMORY_LIMIT,
    domain=domain,
    problem=problem,
    conf_file=CONF_FILE))
condor_script_file.close()
os.chmod(condor_script_path, 0o775)

condor_spec_file = open(os.path.join(CONDOR_DIR, CONDOR_SPEC_NAME), 'w')
condor_spec_file.write(CONDOR_SPEC.format(
    condor_script=CONDOR_SCRIPT_NAME,
    condor_dir=CONDOR_DIR,
    condor_out=CONDOR_OUT,
    condor_err=CONDOR_ERR,
    condor_log=CONDOR_LOG,
    n_conf=len(CONFS)))
condor_spec_file.close()

subprocess.check_call(['condor_submit', os.path.join(CONDOR_DIR, CONDOR_SPEC_NAME)])
subprocess.check_call(['condor_wait', os.path.join(CONDOR_DIR, CONDOR_LOG)])

costs = dict()
for i in range(len(CONFS)):
    planner_output = open(os.path.join(CONDOR_DIR, str(i), CONDOR_OUT)).read()
    cost = -1
    try:
        cost = get_cost(planner_output)
    except:
        pass
    costs[CONFS[i] + MEMORY_LIMIT + str(TIME_LIMIT)] = cost

output_file = open(output_file_path, 'w')
output_file.write(str(costs))
output_file.close()


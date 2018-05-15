#!/usr/bin/python3

import subprocess
import sys

conf = 'learning(ff)'

domain = sys.argv[1]
problem = sys.argv[2]
iterations = int(sys.argv[3])


log = open('rl_driver_log.txt', 'w')

for i in range(iterations):
    print(i)
    subprocess.call(['../fast-downward.py', '--build', 'release64',  domain,  problem, '--search',  conf])
    weights = open('weights.txt')
    log.write(weights.read())
    log.write('\n')
    log.flush()
    weights.close()

log.close()

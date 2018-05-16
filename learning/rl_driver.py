#!/usr/bin/python3

import subprocess
import sys

conf = 'learning(ff,learning_rate=%f)'
learning_rate = 0.1

domain = sys.argv[1]
problem = sys.argv[2]
iterations = int(sys.argv[3])


log = open('rl_driver_log.txt', 'w')

for i in range(1,iterations+1):
    print(i)
    subprocess.call(['../fast-downward.py', '--build', 'release64',  domain,  problem, '--search',  conf % learning_rate])
    weights = open('weights.txt')
    log.write(weights.read())
    log.write('\n')
    log.flush()
    weights.close()
    if i % 100 == 0:
        learning_rate /= 2

log.close()

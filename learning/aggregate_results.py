#!/usr/bin/python3

# Usage: ./aggregate_results.py <results directory> <CSV output file>

import os
import re
import sys


class Result:
    def __init__(self, solved, costs):
        self.solved = solved
        self.costs = costs
        self.avg_cost = 0.0
        self.ipc_score = 0.0

def extract_results(filename):
    f = open(filename)
    print(filename)
    file_content = f.read()
    f.close()
    
    m = re.search('Solved [0-9]+ out of [0-9]+', file_content)
    m2 = re.search('[0-9]+', m.group(0))
    n_solved = int(m2.group(0))
    
    m = re.search('PLAN COSTS:\n(([0-9]+|F) )+', file_content)
    m2 = re.search('(([0-9]+|F) )+', m.group(0))
    costs = m2.group(0).split()
    
    return Result(n_solved, costs)

def numeric_ids(entries):
    result = set()
    for (e, i) in zip(entries, range(len(entries))):
        if e.isnumeric():
            result.add(i)
    return result

def numeric_costs(entries):
    result = []
    for e in entries:
        if e.isnumeric():
            result.append(float(e))
        else:
            result.append(float('inf'))
    return result

def min_lists(l1, l2):
    return [ min(x) for x in zip(l1,l2) ]
        

result_dir = sys.argv[1]
aggregate_file = sys.argv[2]

# Fill a dictrionary of results indexed by configuration and domain

all_domains = set()
all_results = dict()

for conf in os.listdir(result_dir):
    conf_results = dict()
    for result_file in os.listdir(os.path.join(result_dir, conf)):
        domain_name = result_file.replace('.txt','')
        all_domains.add(domain_name)
        conf_results[domain_name] = extract_results(os.path.join(result_dir, conf, result_file))
    all_results[conf] = conf_results


# Find probelms solved by all configurations and compute average plan quality and IPC score

for domain in all_domains:
    # Find the problems and cheapest plans
    id_set = None
    lowest_costs = None
    for conf in all_results:
        costs = all_results[conf][domain].costs
        conf_id_set = numeric_ids(costs)
        if id_set is None:
            id_set = conf_id_set
            lowest_costs = numeric_costs(costs)
        else:
            id_set.intersection_update(conf_id_set)
            lowest_costs = min_lists(lowest_costs, numeric_costs(costs))
    # Compute the average cost and IPC score
    for conf in all_results:
        costs = numeric_costs(all_results[conf][domain].costs)
        cost_sum = 0
        for i in id_set:
            cost_sum += costs[i]
        all_results[conf][domain].avg_cost = cost_sum / len(id_set)
        partial_scores = [ x/y if x != float('inf') else 0.0
            for (x,y) in zip(lowest_costs, costs) ]
        all_results[conf][domain].ipc_score = sum(partial_scores)


# Generate a CSV file

domains = sorted(list(all_domains))
configurations = sorted(all_results.keys())

of = open(aggregate_file, 'w')

of.write(',')
of.write(','.join(configurations))
of.write(',\n')

g_solved = lambda x: x.solved
g_cost = lambda x: round(x.avg_cost, 2)
g_score = lambda x: round(x.ipc_score, 2)

for getter in [g_solved, g_cost, g_score]:
    # Coverage
    for domain in domains:
        of.write(domain)
        of.write(',')
        for conf in configurations:
            if domain in all_results[conf]:
                of.write(str(getter(all_results[conf][domain])))
                of.write(',')
            else:
                of.write('-,')
        of.write('\n')
    of.write('\n')

of.close()


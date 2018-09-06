#!/usr/bin/python3

# Usage: ./aggregate_results.py <results directory> <CSV output file> <plot directory>

import os
import re
import sys

import matplotlib.pyplot as plt

TIMEOUT = 5.0

class Result:
    def __init__(self, solved, costs, times):
        self.solved = solved
        self.costs = costs
        self.times = times
        self.avg_cost = 0.0
        self.ipc_score = 0.0
        self.ipc2_score = 0.0
        self.time_score = sum([(TIMEOUT - x) / TIMEOUT for x in times])

def extract_results(filename):
    f = open(filename)
    print(filename)
    file_content = f.read()
    f.close()
    
    m = re.search('Solved [0-9]+ out of [0-9]+', file_content)
    m2 = re.search('[0-9]+', m.group(0))
    n_solved = int(m2.group(0))
    
    m = re.search('PLAN COSTS:\n(([0-9]+|F) )+([0-9]+|F)', file_content)
    m2 = re.search('(([0-9]+|F) )+([0-9]+|F)', m.group(0))
    costs = m2.group(0).split()
    
    m = re.search('SEARCH TIMES:\n((([0-9\.]+s)|F) )+(([0-9\.]+s)|F)', file_content)
    times_string = m.group(0).replace('SEARCH TIMES:\n','').replace('s', '').replace('F', '')
    times = [float(x) for x in times_string.split()]
    
    return Result(n_solved, costs, times)

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
plotting = False
if len(sys.argv) > 3:
    plotting = True
    plot_dir = sys.argv[3]

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

all_times = dict()
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
        if len(id_set) > 0:
            all_results[conf][domain].avg_cost = cost_sum / len(id_set)
        else:
            all_results[conf][domain].avg_cost = -1
        partial_scores = [ x/y if x != float('inf') else 0.0
            for (x,y) in zip(lowest_costs, costs) ]
        all_results[conf][domain].ipc_score = sum(partial_scores)
        all_results[conf][domain].ipc2_score = sum([x*x for x in partial_scores])
    if plotting:
    # Create the progress plots
        for conf in all_results:
            result = all_results[conf][domain]
            times = sorted(result.times)
            if conf in all_times:
                all_times[conf] += times
            else:
                all_times[conf] = times
            x = [0.0] + times + [TIMEOUT]
            y = list(range(len(times))) + 2 * [len(times)]
            plt.step(x, y, where='post', label=conf)
        plt.legend(loc='upper left', bbox_to_anchor=(1,1))
        plt.savefig(os.path.join(plot_dir, domain), bbox_inches='tight')
        plt.clf()

if plotting:
    # Create the combined progress plot
    for conf in all_times:
        times = sorted(all_times[conf])
        x = [0.0] + times + [TIMEOUT]
        y = list(range(len(times))) + 2 * [len(times)]
        plt.step(x, y, where='post', label=conf)
    plt.ylabel('problems solved')
    plt.xlabel('time [s]')
    plt.legend(loc='lower right', bbox_to_anchor=(1,0)) # 'upper left', (1,1)
    plt.savefig(os.path.join(plot_dir, 'combined'), bbox_inches='tight', format="svg")
    plt.clf()

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
g_score2 = lambda x: round(x.ipc2_score, 2)
g_time_score = lambda x: round(x.time_score, 2)

for getter in [g_solved, g_cost, g_score, g_score2, g_time_score]:
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


import os
import re
import sys

class Result:
    def __init__(self, solved):
        self.solved = solved

def extract_results(filename):
    f = open(filename)
    print filename
    m = re.search('Solved [0-9]+ out of [0-9]+', f.read())
    f.close()
    m2 = re.search('[0-9]+', m.group(0))
    return Result(int(m2.group(0)))

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


# Generate a CSV file

domains = sorted(list(all_domains))
configurations = sorted(all_results.keys())

of = open(aggregate_file, 'w')

of.write(',')
of.write(','.join(configurations))
of.write(',\n')
for domain in domains:
    of.write(domain)
    of.write(',')
    for conf in configurations:
        if all_results[conf].has_key(domain):
            of.write(str(all_results[conf][domain].solved))
            of.write(',')
        else:
            of.write('-,')
    of.write('\n')

of.close()

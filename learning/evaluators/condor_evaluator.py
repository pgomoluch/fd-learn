import os
import numpy as np
import subprocess
import time

from rl_common import get_cost, save_params
from .base_evaluator import BaseEvaluator

CONDOR_DIR = 'condor'
PARAMS_PATH = '../params.txt'
THREADS = 4
ONLINE_REFERENCE_COSTS = True
RETRY_DELAY = 1800


CONDOR_SCRIPT = """#!/bin/bash
cd condor/$1/$2
echo "Hello condor worker 2"
/usr/bin/python {fd_path} --build release64 --overall-time-limit {time_limit} --overall-memory-limit 4G {domain} $3 --heuristic "{heuristic}" --search "{search}" > fd.out
echo "Condor worker done"
"""

CONDOR_PAR_SCRIPT = """#!/bin/bash
echo "Hello condor_par.sh"
pwd
n_batches=$(({n_problems}/{threads}))
param_id=$(($1 / n_batches))
batch_id=$(($1 % n_batches))
skip=$((batch_id * {threads}))
echo "n_batches:" $n_batches "  param_id:" $param_id "  skip:" $skip
cat condor/problems${{batch_id}}.txt

c=0
while read problem; do
    echo "Second arg: " $((skip+c))
    {condor_script} $param_id $((skip+c)) $problem &
    pid[$c]=$!
    ((c++))
done < condor/problems${{batch_id}}.txt
for p in ${{pid[*]}}
do
    wait $p
done
"""

CONDOR_SPEC = """universe = vanilla
executable = {condor_script}
output = condor/condor$(Process).out
error = condor/condor$(Process).err
log = condor/condor.log
arguments = $(Process)
requirements = regexp("^(edge|point|sprite)[0-9][0-9]", TARGET.Machine)
queue {n_jobs}
"""

### requirements = (regexp("^(edge|point|sprite)[0-9][0-9]", TARGET.Machine)) && (Machine != "point42.doc.ic.ac.uk") && (Machine != "sprite14.doc.ic.ac.uk")

class CondorEvaluator(BaseEvaluator):

    def __init__(self, population_size, n_test_problems, domain_path,
        heuristic_str, search_str, max_problem_time, param_handler):
    
        super().__init__(population_size, n_test_problems, domain_path,
            heuristic_str, search_str, max_problem_time, param_handler)
    
        if not os.path.exists(CONDOR_DIR):
            os.makedirs(CONDOR_DIR)
        for h in range (self._population_size):
            for i in range(self._n_test_problems):
                path = os.path.join(CONDOR_DIR, str(h), str(i))
                if not os.path.exists(path):
                    os.makedirs(path)
        
        # Condor script
        condor_script_path = os.path.join(CONDOR_DIR, 'condor.sh')
        condor_par_script_path = os.path.join(CONDOR_DIR, 'condor_par.sh')
        
        condor_file = open(condor_script_path, 'w')
        condor_file.write(CONDOR_SCRIPT.format(
            fd_path=os.path.abspath('../fast-downward.py'),
            time_limit=int(max_problem_time),
            domain=os.path.abspath(domain_path),
            heuristic=heuristic_str,
            search=search_str % PARAMS_PATH))
        condor_file.close()
        
        # Condor parallel script
        condor_par_file = open(condor_par_script_path, 'w')
        condor_par_file.write(CONDOR_PAR_SCRIPT.format(
            threads=THREADS,
            n_problems=self._n_test_problems,
            condor_script=os.path.join('.', CONDOR_DIR, 'condor.sh')))
        condor_par_file.close()
        
        os.chmod(condor_script_path, 0o775)
        os.chmod(condor_par_script_path, 0o775)


    def score_params(self, all_params, paths_and_costs, log=None):
        problem_list = [os.path.abspath(p) for (p, _) in paths_and_costs]
        for params_id, params in enumerate(all_params):
            self._param_handler.save_params(params, os.path.join(CONDOR_DIR, str(params_id), 'params.txt'))
            #problem_list_file = open(os.path.join(CONDOR_DIR, str(params_id), 'problems.txt'),'w')
            #problem_list_file.write(problem_list)
            #problem_list_file.close()
        
        assert len(problem_list) % THREADS == 0
        
        for i in range(len(problem_list) // THREADS):
            batch_problem_list = []
            for j in range(THREADS):
                batch_problem_list.append(problem_list[i * THREADS + j])
            problem_list_file = open(os.path.join(CONDOR_DIR, 'problems%d.txt' % i),'w')
            problem_list_file.write(os.linesep.join(batch_problem_list)+os.linesep)
            problem_list_file.close()
        
        n_jobs = self._population_size * len(problem_list) // THREADS

        condor_spec = CONDOR_SPEC.format(
            condor_script=os.path.abspath('condor/condor_par.sh'),
            n_jobs=n_jobs)
        condor_file = open('condor/condor.cmd', 'w')
        condor_file.write(condor_spec)
        condor_file.close()
        start_time = time.time()
        submit_output = CondorEvaluator._check_output_retry(['condor_submit', 'condor/condor.cmd'], RETRY_DELAY).decode('utf-8')
        cluster_id = CondorEvaluator._get_condor_cluster(submit_output)
        
        print('Waiting for condor...')
        subprocess.check_call(['condor_wait', 'condor/condor.log'])
        # Discard the last 10% of jobs
        #CondorEvaluator._check_call_retry(['condor_wait', 'condor/condor.log', str(cluster_id), '-num', str(int(0.9 * n_jobs))], RETRY_DELAY)
        #try:
        #    subprocess.check_call(['condor_rm', 'cluster', str(cluster_id)])
        #except:
        #    print('condor_rm failed')
        os.remove('condor/condor.log') # a workaround for the new version of Condor
        elapsed_time = time.time() - start_time
        print('{} jobs ({} runs) completed in {} s.'.format(
            n_jobs, self._population_size * self._n_test_problems,
            round(elapsed_time,2)))
        if log:
            log.write(str(round(elapsed_time,2)) + '\n')
            log.flush()
        
        # Aggregate the results
        aggregation_start = time.time()
        iteration_costs = np.zeros([self._population_size, self._n_test_problems])
        for params_id in range(self._population_size):
            for problem_id, (_, ref_cost) in enumerate(paths_and_costs):
                # Determine task completion in current iteration, based on sas_plan existence
                sas_plan_path = 'condor/{}/{}/sas_plan'.format(params_id,problem_id)
                if not os.path.exists(sas_plan_path):
                    iteration_costs[params_id, problem_id] = -1
                    continue
                os.remove(sas_plan_path)
                
                output_path = 'condor/{}/{}/fd.out'.format(params_id,problem_id)
                output_file = open(output_path)
                planner_output = output_file.read()
                output_file.close()
                
                plan_cost = -1
                try:
                    plan_cost = get_cost(planner_output)
                except:
                    pass
                iteration_costs[params_id, problem_id] = plan_cost
        
        
        if log:
            log.write('Iteration costs:\n')
            for row in iteration_costs:
                for entry in row:
                    log.write('%6d' % entry)
                log.write('\n')
            log.write('\n')
        
        
        if ONLINE_REFERENCE_COSTS:    
            old_costs = [ c for (_, c) in paths_and_costs ]
            reference_costs = self.get_online_reference_costs(old_costs, iteration_costs)
        else:    
            (_, reference_costs) = paths_and_costs
        
        
        if log:
            log.write('Reference costs:\n')
            for entry in reference_costs:
                log.write('%6d' % entry)
            log.write('\n\n')
        
        
        total_scores = self.compute_total_scores(iteration_costs, reference_costs)
        
        print('Aggregation of the results took {} s.'.format(round(time.time()-aggregation_start)))
        return total_scores
    
    
    @staticmethod
    def _get_condor_cluster(submit_output):
        #TODO find a better way to get this number
        space = submit_output.rfind(' ')
        dot = submit_output.rfind('.')
        return int(submit_output[space+1 : dot])
    
    @staticmethod
    def _check_call_retry(command, delay):
        while True:
            try:
                subprocess.check_call(command)
                return
            except subprocess.CalledProcessError as e:
                print('check_call failed')
                print(e)
                time.sleep(delay)
                print('retrying...')

    @staticmethod
    def _check_output_retry(command, delay):
        while True:
            try:
                return subprocess.check_output(command)
            except subprocess.CalledProcessError as e:
                print('check_output failed')
                print(e)
                time.sleep(delay)
                print('retrying...') 
                        


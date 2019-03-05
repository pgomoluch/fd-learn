import os
import numpy as np
import subprocess
import time

from rl_common import get_cost, compute_ipc_reward, save_params


CONDOR_DIR = 'condor'
PARAMS_PATH = '../params.txt'
THREADS = 4
ONLINE_REFERENCE_COSTS = True
PROBLEM_PARALLEL = True

CONDOR_SCRIPT = """#!/bin/bash
cd condor/$1
i=0
while read problem; do
    cd $i
    /usr/bin/python {fd_path} --build release64 --overall-time-limit {time_limit} {domain} $problem --heuristic "{heuristic}" --search "{search}" > fd.out
    cd ..
    ((i++))
done < problems.txt
"""

CONDOR_PAR_SCRIPT = """#!/bin/bash
start=$(($1*{threads}))
for ((i=0; i<{threads}; i++))
do
    {condor_script} $((start+i)) &
    pid[$i]=$!
done

for p in ${{pid[*]}}
do
    wait $p
done
"""

DP_CONDOR_SCRIPT = """#!/bin/bash
cd condor/$1/$2
echo "Hello condor worker 2"
/usr/bin/python {fd_path} --build release64 --overall-time-limit {time_limit} {domain} $3 --heuristic "{heuristic}" --search "{search}" > fd.out
echo "Condor worker done"
"""

DP_CONDOR_PAR_SCRIPT = """#!/bin/bash
echo "Hello condor_par.sh"
pwd
cat condor/$1/problems.txt
c=0
while read problem; do
    {condor_script} $1 $c $problem &
    pid[$i]=$!
    ((c++))
done < condor/$1/problems.txt
for p in ${{pid[*]}}
do
    wait $p
done
"""

CONDOR_SPEC = """universe = vanilla
executable = {condor_script}
output = condor/$(Process)/condor.out
error = condor/$(Process)/condor.err
log = condor/condor.log
arguments = $(Process)
requirements = regexp("^(edge|point|sprite)[0-9][0-9]", TARGET.Machine)
queue {population_size}
"""


class CondorEvaluator:

    def __init__(self, target_types, population_size, n_test_problems,
        domain_path, heuristic_str, search_str, max_problem_time):
    
        self._target_types = target_types
        self._population_size = population_size
        self._n_test_problems = n_test_problems
    
        if not os.path.exists(CONDOR_DIR):
            os.makedirs(CONDOR_DIR)
        for h in range (self._population_size):
            for i in range(self._n_test_problems):
                path = os.path.join(CONDOR_DIR, str(h), str(i))
                if not os.path.exists(path):
                    os.makedirs(path)
        
        # Condor scripts
        condor_script_path = os.path.join(CONDOR_DIR, 'condor.sh')
        condor_par_script_path = os.path.join(CONDOR_DIR, 'condor_par.sh')
        
        if PROBLEM_PARALLEL:
            SCRIPT = DP_CONDOR_SCRIPT
            PAR_SCRIPT = DP_CONDOR_PAR_SCRIPT
        else:
            SCRIPT = CONDOR_SCRIPT
            PAR_SCRIPT = CONDOR_PAR_SCRIPT
        
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
            condor_script=os.path.join('.', CONDOR_DIR, 'condor.sh')))
        condor_par_file.close()
        
        os.chmod(condor_script_path, 0o775)
        os.chmod(condor_par_script_path, 0o775)


    def score_params(self, all_params, paths_and_costs, log=None):
        problem_list = os.linesep.join([os.path.abspath(p) for (p, _) in paths_and_costs]) + os.linesep
        for params_id, params in enumerate(all_params):
            save_params(params, self._target_types, os.path.join(CONDOR_DIR, str(params_id), 'params.txt'))
            problem_list_file = open(os.path.join(CONDOR_DIR, str(params_id), 'problems.txt'),'w')
            problem_list_file.write(problem_list)
            problem_list_file.close()
        
        spec_population_size = self._population_size if PROBLEM_PARALLEL else self._population_size // THREADS
        condor_spec = CONDOR_SPEC.format(
            condor_script=os.path.abspath('condor/condor_par.sh'),
            population_size=spec_population_size)
        condor_file = open('condor/condor.cmd', 'w')
        condor_file.write(condor_spec)
        condor_file.close()
        start_time = time.time()
        submit_output = subprocess.check_output(['condor_submit', 'condor/condor.cmd']).decode('utf-8')
        cluster_id = CondorEvaluator._get_condor_cluster(submit_output)
        
        print('Waiting for condor...')
        #subprocess.check_call(['condor_wait', 'condor/condor.log'])
        # Discard the last 10% of jobs
        subprocess.check_call(['condor_wait', 'condor/condor.log', str(cluster_id), '-num', str(int(0.9 * spec_population_size))])
        try:
            subprocess.check_call(['condor_rm', 'cluster', str(cluster_id)])
        except:
            print('condor_rm failed')
        elapsed_time = time.time() - start_time
        print('{} jobs ({} runs) completed in {} s.'.format(
            self._population_size/THREADS, self._population_size * self._n_test_problems,
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
        if ONLINE_REFERENCE_COSTS:
            # Update the reference costs using the costs from this iteration
            print(iteration_costs)
            reference_costs = []
            for (_, old_ref), costs in zip(paths_and_costs, iteration_costs.T):
                actual_costs = [c for c in costs if c > 0.0] 
                if actual_costs:
                    if old_ref > 0.0:
                        new_ref = min(old_ref, min(actual_costs))
                    else:
                        new_ref = min(actual_costs)
                else:
                    new_ref = old_ref
                reference_costs.append(new_ref)
        else:
            (_, reference_costs) = paths_and_costs 
        print(reference_costs)       
        
        # Compute the scores
        total_scores = []
        for params_id in range(self._population_size):
            total_score = 0.0
            for problem_id, ref_cost in enumerate(reference_costs):
                try:
                    reward = compute_ipc_reward(iteration_costs[params_id, problem_id], ref_cost)
                except:
                    reward = 0.0
                total_score += reward
            total_scores.append(total_score)
        
        print('Aggregation of the results took {} s.'.format(round(time.time()-aggregation_start)))
        return total_scores
    
    @staticmethod
    def _get_condor_cluster(submit_output):
        #TODO find a better way to get this number
        space = submit_output.rfind(' ')
        dot = submit_output.rfind('.')
        return int(submit_output[space+1 : dot])


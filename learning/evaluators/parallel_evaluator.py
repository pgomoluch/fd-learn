import itertools
import os
import numpy as np
import subprocess
import time

from multiprocessing import Pool

from rl_common import get_cost, save_params
from .base_evaluator import BaseEvaluator

N_PROC = 4
PARAMS_PATH = 'params{}.txt'
ONLINE_REFERENCE_COSTS = True


class ParallelEvaluator(BaseEvaluator):

    def __init__(self, population_size, n_test_problems, domain_path,
        heuristic_str, search_str, max_problem_time, param_handler):
    
        super().__init__(population_size, n_test_problems, domain_path,
            heuristic_str, search_str, max_problem_time, param_handler)
        
        self.fd_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'fast-downward.py'
        )
        for i in range(population_size):
            for j in range(n_test_problems):
                os.makedirs(os.path.join(str(i), str(j)), exist_ok=True)


    def score_params(self, all_params, paths_and_costs, log=None):
        
        problem_list = [os.path.abspath(p) for (p, _) in paths_and_costs]
        iteration_costs = np.zeros([self._population_size, self._n_test_problems])
        
        for params_id, params in enumerate(all_params):
            self._param_handler.save_params(params, PARAMS_PATH.format(params_id))
            
        for i in range(len(problem_list)):
            problem_list[i] = (i, problem_list[i]) # add ids to problem paths
        print(problem_list)    
        
        # get a cross product
        params_problems = itertools.product(range(len(all_params)), problem_list)
        
        start_time = time.time()
        
        with Pool(N_PROC) as p:
            iteration_costs = p.map(self.run_planner, params_problems)
        
        elapsed_time = time.time() - start_time
        
        if log:
            log.write(str(round(elapsed_time,2)) + '\n')
            log.flush()
        
        iteration_costs = np.reshape(np.array(iteration_costs), (self._population_size, self._n_test_problems))
        
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
        
        return total_scores

    def run_planner(self, params_problem):
        params_id, (problem_id, problem_path) = params_problem
        
        try:
            planner_output = subprocess.check_output([str(self.fd_path), '--build', 'release64',
                '--overall-time-limit', str(int(self._max_problem_time)),
                '--overall-memory-limit', '4G', os.path.abspath(self._domain_path), problem_path, '--heuristic',
                self._heuristic_str, '--search',
                self._search_str % ('../../' + PARAMS_PATH.format(params_id))],
                cwd=os.path.join(str(params_id), str(problem_id))
            ).decode('utf-8')
            try:
                plan_cost = get_cost(planner_output)
            except:
                planner_output = 'Error getting cost.'
                plan_cost = -1
        except subprocess.CalledProcessError as e:
            planner_output = 'CalledProcessError:\n' + e.output.decode('utf-8')
            plan_cost = -1
        
        output_file = open(os.path.join(str(params_id), str(problem_id), 'out.txt'), 'w')
        output_file.write(planner_output)
        output_file.close()
        
        return plan_cost


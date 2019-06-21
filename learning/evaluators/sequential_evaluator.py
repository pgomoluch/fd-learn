import os
import numpy as np
import subprocess
import time

from rl_common import get_cost, save_params
from .base_evaluator import BaseEvaluator

PARAMS_PATH = 'params-se.txt'
ONLINE_REFERENCE_COSTS = True


class SequentialEvaluator(BaseEvaluator):

    def __init__(self, population_size, n_test_problems, domain_path,
        heuristic_str, search_str, max_problem_time, param_handler):
    
        super().__init__(population_size, n_test_problems, domain_path,
            heuristic_str, search_str % PARAMS_PATH, max_problem_time, param_handler)


    def score_params(self, all_params, paths_and_costs, log=None):
        
        problem_list = [os.path.abspath(p) for (p, _) in paths_and_costs]
        iteration_costs = np.zeros([self._population_size, self._n_test_problems])
        
        for params_id, params in enumerate(all_params):
            self._param_handler.save_params(params, PARAMS_PATH)
            for problem_id, problem_path in enumerate(problem_list):
                
                try:
                    planner_output = subprocess.check_output(['../fast-downward.py', '--build', 'release64',
                        '--overall-time-limit', str(int(self._max_problem_time)),
                        '--overall-memory-limit', '4G', self._domain_path, problem_path, '--heuristic',
                        self._heuristic_str, '--search', self._search_str])
                    try:
                        plan_cost = get_cost(planner_output)
                    except:
                        plan_cost = -1
                except subprocess.CalledProcessError as e:
                    plan_cost = -1
                
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
        
        return total_scores


import errno
import itertools
import os
import numpy as np
import subprocess
import shutil
import time

from contextlib import closing
from multiprocessing import Pool

from rl_common import get_cost, save_params
from .base_evaluator import BaseEvaluator

N_PROC = 4
PARAMS_PATH = 'params{}.txt'
PROBLEM_PATH = 'problem{}.pddl'
ITERATION_COSTS_PATH = 'costs.npz'
ONLINE_REFERENCE_COSTS = True


class MPIEvaluator(BaseEvaluator):

    def __init__(self, population_size, n_test_problems, domain_path,
        heuristic_str, search_str, max_problem_time, param_handler):
    
        super(MPIEvaluator, self).__init__(population_size, n_test_problems, domain_path,
            heuristic_str, search_str, max_problem_time, param_handler)
        
        self.fd_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'fast-downward.py'
        )
        for i in range(population_size):
            for j in range(n_test_problems):
                #os.makedirs(os.path.join(str(i), str(j)), exist_ok=True) # python3
                try:
                    os.makedirs(os.path.join(str(i), str(j)))
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise


    def score_params(self, all_params, paths_and_costs, log=None):
        
        problem_list = [os.path.abspath(p) for (p, _) in paths_and_costs]
        iteration_costs = np.zeros([self._population_size, self._n_test_problems])
        
        for params_id, params in enumerate(all_params):
            self._param_handler.save_params(params, PARAMS_PATH.format(params_id))
            
        for i in range(len(problem_list)):
            shutil.copyfile(problem_list[i], PROBLEM_PATH.format(i))

        mpi_runner_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            'mpi_runner.py')
        fd_command = [self.fd_path, '--build', 'release64',
            '--overall-time-limit', str(int(self._max_problem_time)),
            '--overall-memory-limit', '4G', os.path.abspath(self._domain_path), 'notused',
            '--heuristic', self._heuristic_str, '--search', self._search_str]

        start_time = time.time()
        
        subprocess.check_call(['mpiexec', '-n', '1', '-usize', str(N_PROC+1),
            'python', mpi_runner_path, str(len(all_params)), str(len(paths_and_costs)),
            ITERATION_COSTS_PATH] + fd_command)
        
        elapsed_time = time.time() - start_time
        
        iteration_costs = np.load(ITERATION_COSTS_PATH)
        
        if log:
            log.write(str(round(elapsed_time,2)) + '\n')
            log.write('Iteration costs:\n')
            for row in iteration_costs:
                for entry in row:
                    log.write('%6d' % entry)
                log.write('\n')
            log.write('\n')
            log.flush()
        
        
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


from mpi4py.futures import MPIPoolExecutor

import itertools
import numpy as np
import os
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_common import get_cost

PROBLEM_PATH_ID = 8
PARAM_PATH_ID = 12

LOCAL_PROBLEM_NAME = 'problem{}.pddl'
LOCAL_PARAMS_NAME = 'params{}.txt'

def run_planner(data):
    
    (params_id, problem_id, fd_command) = data
    
    fd_command[PROBLEM_PATH_ID] = os.path.abspath(LOCAL_PROBLEM_NAME.format(problem_id))
    fd_command[PARAM_PATH_ID] = fd_command[PARAM_PATH_ID] % (
        os.path.abspath(LOCAL_PARAMS_NAME.format(params_id)))
    
    try:
        planner_output = subprocess.check_output(fd_command,
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


if __name__ == '__main__':

    n_params = int(sys.argv[1])
    n_problems = int(sys.argv[2])
    result_path = sys.argv[3]
    fd_command = sys.argv[4:]
    
    pp = itertools.product(range(n_params), range(n_problems))
    data = [ p + (fd_command,) for p in pp ]
    
    with MPIPoolExecutor() as executor:
        iteration_costs = executor.map(run_planner, data)
    
    iteration_costs = np.reshape(np.array(list(iteration_costs)), (n_params, n_problems))
    result_file = open(result_path, 'wb')
    
    np.save(result_file, iteration_costs)
    result_file.close()


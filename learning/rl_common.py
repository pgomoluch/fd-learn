import psutil
import re
import signal
import subprocess


def get_output_with_timeout(command, timeout):

    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    try:
        output = proc.communicate(timeout=timeout)[0]
        return output
    except subprocess.TimeoutExpired:
        try:
            parent = psutil.Process(proc.pid)
            children = parent.children(recursive=True)
            for c in children:
                try:
                    c.send_signal(signal.SIGINT)
                except psutil.NoSuchProcess:
                    pass
            proc.kill()
        except psutil.NoSuchProcess:
            pass
        raise

def get_cost(planner_output):
    if not isinstance(planner_output, str):
        planner_output = planner_output.decode('utf-8')
    m_iter = re.finditer('(Plan cost: )([0-9]+)', planner_output)
    cost = min([int(m.group(2)) for m in m_iter]) 
    return cost

def compute_reference_costs(domain_path, problem_path, ref_confs, heuristic, timeout):
    costs = []
    for ref_search in ref_confs:
        try:
            reference_output = get_output_with_timeout(['../fast-downward.py',
                '--build', 'release64',  domain_path,  problem_path,
                '--heuristic', heuristic, '--search',  ref_search], timeout)
            cost = get_cost(reference_output)
            print('Reference cost:', cost)
            costs.append(cost)
        except subprocess.TimeoutExpired:
            print('Failed to find a reference solution.')
    return costs

class NoRewardException(Exception):
    pass

def compute_ipc_reward(plan_cost, reference_cost):
    # If no reference solution
    if reference_cost == -1:
        if plan_cost > 0:
            return 2.0
        raise NoRewardException()
    # Otherwise
    if plan_cost < 0:
        return 0.0
    elif plan_cost == 0:
        raise NoRewardException()
    else:
        return reference_cost / plan_cost

def save_params(params, target_types, filename):
    f = open(filename, 'w')
    for (p, t) in zip(params, target_types):
        if t == int:
            p = int(round(p))
        f.write(str(p))
        f.write('\n')
    f.close()

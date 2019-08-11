from rl_common import compute_ipc_reward, NoRewardException

class BaseEvaluator(object):
    
    def __init__(self, population_size, n_test_problems, domain_path,
        heuristic_str, search_str, max_problem_time, param_handler):
        
        self._population_size = population_size
        self._n_test_problems = n_test_problems
        self._domain_path = domain_path
        self._heuristic_str = heuristic_str
        self._search_str = search_str
        self._max_problem_time = max_problem_time
        self._param_handler = param_handler
    
    def get_online_reference_costs(self, old_costs, iteration_costs):
        # Update the reference costs using the costs from this iteration
        reference_costs = []
        for old_ref, costs in zip(old_costs, iteration_costs.T):
            actual_costs = [c for c in costs if c > 0.0] 
            if actual_costs:
                if old_ref > 0.0:
                    new_ref = min(old_ref, min(actual_costs))
                else:
                    new_ref = min(actual_costs)
            else:
                new_ref = old_ref
            reference_costs.append(new_ref)

        return reference_costs
    
    
    def compute_total_scores(self, iteration_costs, reference_costs):
        total_scores = []
        for params_id in range(self._population_size):
            total_score = 0.0
            for problem_id, ref_cost in enumerate(reference_costs):
                try:
                    reward = compute_ipc_reward(iteration_costs[params_id, problem_id], ref_cost)
                except NoRewardException:
                    reward = 0.0
                total_score += reward
            total_scores.append(total_score)
        
        return total_scores

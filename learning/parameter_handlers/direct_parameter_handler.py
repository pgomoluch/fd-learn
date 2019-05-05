import numpy as np

class DirectParameterHandler:
    
    def __init__(self):
        # [epsilon, stall_size, number of random walks, random_walk_length,
        # cycle_length, fraction_local]
        self.initial_mean = [0.5, 10.0, 5.0, 10.0, 200.0, 0.5]
        self.initial_stddev = [0.5, 10.0, 5.0, 10.0, 200.0, 0.5]
        self.min_params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.max_params = [1.0, float('inf'), float('inf'), float('inf'), float('inf'), 1.0]
        self.target_types = [float, int, int, int, int, float]
        
    def save_params(self, params, filename):
        f = open(filename, 'w')
        for (p, t) in zip(params, self.target_types):
            if t == int:
                p = int(round(p))
            f.write(str(p))
            f.write('\n')
        f.close()
    
    def bound_params(self, params):
        for i in range(len(params)):
            if params[i] < self.min_params[i]:
                params[i] = self.min_params[i]
            elif params[i] > self.max_params[i]:
                params[i] = self.max_params[i]
        

import numpy as np

class NeuralParameterHandler:
    
    N_FEATURES = 4
    N_SEARCH_PARAMETERS = 6
    
    def __init__(self):
        # no hidden layers for now
        self.shape = (self.N_FEATURES, self.N_SEARCH_PARAMETERS)
        n_net_params = (self.N_FEATURES * self.N_SEARCH_PARAMETERS +
            self.N_SEARCH_PARAMETERS)
        self.initial_mean = [0.0] * n_net_params
        self.initial_stddev = [0.1] * n_net_params
        
    def save_params(self, params, filename):
        f = open(filename, 'w')
        f.write(' '.join([str(x) for x in self.shape]))
        f.write('\n')
        f.write(' '.join([str(x) for x in params]))
        f.close()
    
    def bound_params(self, params):
        pass
        

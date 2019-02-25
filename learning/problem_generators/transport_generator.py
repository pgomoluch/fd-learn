import os
import time
import subprocess


#ipc_generator = '../../../IPC/own-transport/generator14L/city-generator.py'
#ipc_generator = '../../../IPC/own-transport/generator14L/two-cities-generator.py'
ipc_generator = '../../../IPC/own-transport/generator14L/three-cities-generator.py'


class TransportGenerator:
    
    def __init__(self, trucks, packages):
        self.trucks = trucks
        self.packages = packages
        
        self.nodes = 15
        self.size = 1000
        self.degree = 4
        self.mindistance = 100
    
    def generate(self, result_path = 'problem.pddl'):
        seed = time.time()
        ipc_generator_command = ['python2', ipc_generator, str(self.nodes), str(self.size),
            str(self.degree), str(self.mindistance), str(self.trucks), str(self.packages), str(seed)]
        problem = subprocess.check_output(ipc_generator_command).decode('utf-8')
        problem_file = open(result_path, 'w')
        problem_file.write(problem)
        problem_file.close()
        # Remove the tex file created by the generator
        tex_path = 'city-sequential-%dnodes-%dsize-%ddegree-%dmindistance-%dtrucks-%dpackages-%dseed.tex' % (self.nodes, self.size, self.degree, self.mindistance, self.trucks, self.packages, int(seed))
        if os.path.exists(tex_path):
            os.remove(tex_path)
    
    def generate_batch(self, n, base_path = 'problem'):
        for i in range(1, n+1):
            path = base_path + str(i) + '.pddl'
            self.generate(path)
    
    def easier(self):
        if self.packages > 1:
            self.packages -= 1
    
    def harder(self):
        self.packages += 1

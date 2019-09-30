import os
#import pathlib
import time
import subprocess

from .base_generator import BaseGenerator

class TransportGenerator(BaseGenerator):
    
    # ../../../IPC/own-transport/generator14L/ relative to this module; move to configuration file
    #generators_dir = pathlib.Path(*pathlib.Path(__file__).absolute().parts[:-5]).joinpath('IPC', 'own-transport', 'generator14L')
    generator_dir = os.path.abspath(__file__)
    for i in range(5):
        generator_dir = os.path.dirname(generator_dir)
    generator_dir = os.path.join(generator_dir, 'IPC', 'own-transport', 'generator14L')
    
    #ipc_generator1 = str(generators_dir.joinpath('city-generator.py'))
    #ipc_generator2 = str(generators_dir.joinpath('two-cities-generator.py'))
    #ipc_generator3 = str(generators_dir.joinpath('three-cities-generator.py'))
    ipc_generator1 = str(os.path.join(generator_dir, 'city-generator.py'))
    ipc_generator2 = str(os.path.join(generator_dir, 'two-cities-generator.py'))
    ipc_generator3 = str(os.path.join(generator_dir, 'three-cities-generator.py'))
        
    def __init__(self, trucks, packages, nodes=15, cities=1, degree=4):
        self.trucks = trucks
        self.packages = packages
        self.nodes = nodes
        self.degree = degree
        
        self.size = 1000
        self.mindistance = 100
        
        if cities == 1:
            self.generator = self.ipc_generator1
        elif cities == 2:
            self.generator = self.ipc_generator2
        elif cities == 3:
            self.generator = self.ipc_generator3
        else:
            raise ValueError('{} is an invalid number of cities'.format(cities))
    
    def __str__(self):
        return 'TransportGenerator(%d trucks, %d packages, %d nodes)' % (
            self.trucks, self.packages, self.nodes)    
    
    def generate(self, result_path = 'problem.pddl'):
        seed = time.time()
        ipc_generator_command = ['python2', self.generator, str(self.nodes), str(self.size),
            str(self.degree), str(self.mindistance), str(self.trucks), str(self.packages), str(seed)]
        fail_count = 0
        while fail_count < 5:
            try:
                problem = subprocess.check_output(ipc_generator_command).decode('utf-8')
                problem_file = open(result_path, 'w')
                problem_file.write(problem)
                problem_file.close()
                break
            except subprocess.CalledProcessError:
                fail_count += 1
                if fail_count > 4:
                    break
                else:
                    time.sleep(1)
                    seed = time.time()
        # Remove the tex file created by the generator
        tex_path = 'city-sequential-%dnodes-%dsize-%ddegree-%dmindistance-%dtrucks-%dpackages-%dseed.tex' % (self.nodes, self.size, self.degree, self.mindistance, self.trucks, self.packages, int(seed))
        if os.path.exists(tex_path):
            os.remove(tex_path)
    
    def easier(self):
        if self.nodes >= 10 and self.packages >= 3:
            self.nodes -= 5
            self.packages -= 2
    
    def harder(self):
        self.nodes += 5
        self.packages += 2
    
    conf = {
        'ipc2011': ((4,16,40,1), (4,18,45,1), (4,18,18,1), (4,12,18,1), (4,14,21,1),
            (4,16,24,1), (4,18,24,1), (4,20,50,1), (4,22,50,1), (4,20,53,1),
            (4,22,53,1), (4,20,30,2), (4,22,60,2), (4,20,62,2), (4,22,62,2),
            (4,20,20,3), (4,22,60,3), (4,20,63,3), (4,22,63,3), (4,22,66,3)),
        'agr2019': ((4,11,15,1,3),) * 10
    }
    
    #@classmethod
    #def generate_series(cls, key, directory):
    #    for i, c in enumerate(cls.conf[key]):
    #        generator = cls(*c)
    #        generator.generate(os.path.join(directory, '{:02d}.pddl'.format(i+1)))


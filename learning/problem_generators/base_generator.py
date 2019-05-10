import os

class BaseGenerator:
    
    def generate_batch(self, n, base_path = 'problem'):
        for i in range(1, n+1):
            path = base_path + str(i) + '.pddl'
            self.generate(path)
    
    @classmethod
    def generate_series(cls, key, directory):
        for i, c in enumerate(cls.conf[key]):
            generator = cls(*c)
            generator.generate(os.path.join(directory, '{:02d}.pddl'.format(i+1)))

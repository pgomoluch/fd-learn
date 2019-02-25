import os
import time
import shutil
import subprocess

ipc_generator = '../../../IPC/own-spanner/spanner-generator.py'

class SpannerGenerator:

    def __init__(self, spanners, nuts, locations):
        self.spanners = spanners
        self.nuts = nuts
        self.locations = locations
    
    def generate(self, result_path = 'problem.pddl'):
        ipc_generator_command = [ipc_generator, str(self.spanners), str(self.nuts),
            str(self.locations)]
        output = subprocess.check_output(ipc_generator_command).decode('utf-8')
        filename = output[(output.rfind(':')+1):-1]
        shutil.move(filename, result_path)
        time.sleep(1.1)
    
    def generate_batch(self, n, base_path = 'problem'):
        for i in range(1, n+1):
            path = base_path + str(i) + '.pddl'
            self.generate(path)


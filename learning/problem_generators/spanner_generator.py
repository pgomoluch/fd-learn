import os
import time
import shutil
import subprocess

from .base_generator import BaseGenerator

ipc_generator = '../../../IPC/own-spanner/spanner-generator.py'

class SpannerGenerator(BaseGenerator):

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


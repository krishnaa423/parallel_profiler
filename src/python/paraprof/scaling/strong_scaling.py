#region modules
from paraprof.scaling.profiler import Profiler
import os 
import matplotlib.pyplot as plt; plt.style.use('seaborn-v0_8-whitegrid')
import configparser
import shlex 
#endregion

#region variables
#endregion

#region functions
#endregion

#region classes
class StrongScaling(Profiler):
    @property
    def mpi_str(self) -> str:
        assert self.args.program is not None, 'provide --program with program name'
        
        problem_size_str: str = str(0) if self.args.problem_sizes is None else str(self.args.problem_sizes[0])
        nthread_str: str = str(self.args.nthreads[0])

        ntask_fstring = "{ntask}"
        elapsed_time_fstring = "{elapsed_time}"
        empty_dict = '{}'
        
        output: str = f'''#!/usr/bin/env python

from subprocess import run
import time 
import configparser
import os
import shlex

ntasks = {self.args.ntasks}
elapsed_times = []
mpi_prefix_args = ''
cmd_extra_args = ''
start_ntask_idx = 0
stop_ntask_idx = {len(self.args.ntasks)}

# Set omp_num_threads.
os.environ['OMP_NUM_THREADS'] = str({self.args.nthreads[0]})

for ntask in ntasks[start_ntask_idx:stop_ntask_idx]:
    # Build the cmd.
    problem_size_arg = [str({self.args.problem_sizes[0]})]
    cmd = shlex.split('{self.args.mpiexec}') + shlex.split(f'-n {ntask_fstring}') + shlex.split(mpi_prefix_args) + shlex.split('{self.args.program}') + problem_size_arg + shlex.split(cmd_extra_args)

    # Run and time task
    start_time = time.perf_counter()
    print('Running: ', ' '.join(cmd), '.', flush=True)
    result = run(cmd)
    stop_time = time.perf_counter()

    
    # Calculate elapsed time.
    elapsed_time = stop_time - start_time
    print(f'Done in {elapsed_time_fstring} seconds.\\n\\n', flush=True)
    elapsed_times.append(elapsed_time)


filename = 'strong-scaling.ini'
# Create file if it does not exist.
if not os.path.exists(filename):
    os.system('touch ' + filename)

# Read sections in ini file.
config = configparser.ConfigParser()
config.read(filename)

# Update section.
tag = 'mpi-' + str({nthread_str}) + '-' + str({problem_size_str})
if tag not in config.sections(): config[tag] = {empty_dict}
for ntask, elapsed_time in zip(ntasks, elapsed_times):
    config[tag][str(ntask)] = str(elapsed_time)

# Write section.
with open(filename, 'w') as f: config.write(f)
'''
        return output

    @property
    def openmp_str(self) -> str:
        assert self.args.program is not None, 'provide --program with program name'

        problem_size_str: str = str(0) if self.args.problem_sizes is None else str(self.args.problem_sizes[0])
        ntask_str: str = str(self.args.ntasks[0])

        nthread_fstring = "{nthread}"
        elapsed_time_fstring = "{elapsed_time}"
        empty_dict = '{}'
        
        output: str = f'''#!/usr/bin/env python

from subprocess import run
import time 
import configparser
import os
import shlex

nthreads = {self.args.nthreads}
elapsed_times = []
mpi_prefix_args = ''
cmd_extra_args = ''
start_nthread_idx = 0
stop_nthread_idx = {len(self.args.nthreads)}

for nthread in nthreads[start_nthread_idx:stop_nthread_idx]:
    # Build the cmd.
    problem_size_arg = [str({self.args.problem_sizes[0]})]
    cmd = shlex.split('{self.args.mpiexec}') + shlex.split(f'-n {self.args.ntasks[0]}') + shlex.split(mpi_prefix_args) + shlex.split('{self.args.program}') + problem_size_arg + shlex.split(cmd_extra_args)

    # Set omp_num_threads. 
    os.environ['OMP_NUM_THREADS'] = str(f'{nthread_fstring}')

    # Run and time task
    start_time = time.perf_counter()
    print('Running: ', ' '.join(cmd), '.', flush=True)
    result = run(cmd)
    stop_time = time.perf_counter()

    
    # Calculate elapsed time.
    elapsed_time = stop_time - start_time
    print(f'Done in {elapsed_time_fstring} seconds.\\n\\n', flush=True)
    elapsed_times.append(elapsed_time)


filename = 'strong-scaling.ini'
# Create file if it does not exist.
if not os.path.exists(filename):
    os.system('touch ' + filename)

# Read sections in ini file.
config = configparser.ConfigParser()
config.read(filename)

# Update section.
tag = 'openmp-' + str({ntask_str}) + '-' + str({problem_size_str})
if tag not in config.sections(): config[tag] = {empty_dict}
for nthread, elapsed_time in zip(nthreads, elapsed_times):
    config[tag][str(nthread)] = str(elapsed_time)

# Write section.
with open(filename, 'w') as f: config.write(f)
'''
        return output

    @property
    def strong_scaling_str(self) -> str:
        if self.args.mpi:
            return self.mpi_str
        elif self.args.openmp:
            return self.openmp_str
        else:   # Default is mpi mode.
            return self.mpi_str

    @property
    def file_contents(self) -> dict:
        return {
            'strong-scaling.sh': self.strong_scaling_str
        }
    
    def plot(self):
        filename = 'strong-scaling.ini'
        assert os.path.exists(filename), 'strong-scaling.ini file does not exist.'

        config = configparser.ConfigParser()
        config.read(filename)
        
        for tag in config.sections():
            ntasks = []
            elapsed_times = []
            for ntask, elapsed_time in config[tag].items(): ntasks.append(int(ntask)); elapsed_times.append(float(elapsed_time))
            plt.scatter(ntasks, elapsed_times, label=tag)
            plt.plot(ntasks, elapsed_times)
        

        plt.xlabel('Tasks/Threads')
        plt.ylabel('Time elapsed (s)')
        plt.title('Strong scaling')
        plt.legend()
        plt.savefig('strong-scaling.png')
        plt.show()

#endregion
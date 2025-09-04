#!/usr/bin/env python

from subprocess import run
import time 
import configparser
import os
import shlex

ntasks = [1, 2, 4, 8, 16, 32]
elapsed_times = []
mpi_prefix_args = ''
cmd_extra_args = ''
start_ntask_idx = 0
stop_ntask_idx = 6

# Set omp_num_threads.
os.environ['OMP_NUM_THREADS'] = str(1)

for ntask in ntasks[start_ntask_idx:stop_ntask_idx]:
    # Build the cmd.
    problem_size_arg = [str(16384)]
    cmd = shlex.split('srun') + shlex.split(f'-n {ntask}') + shlex.split(mpi_prefix_args) + shlex.split('python ./test.py') + problem_size_arg + shlex.split(cmd_extra_args)

    # Run and time task
    start_time = time.perf_counter()
    print('Running: ', ' '.join(cmd), '.', flush=True)
    result = run(cmd)
    stop_time = time.perf_counter()

    
    # Calculate elapsed time.
    elapsed_time = stop_time - start_time
    print(f'Done in {elapsed_time} seconds.\n\n', flush=True)
    elapsed_times.append(elapsed_time)


filename = 'strong-scaling.ini'
# Create file if it does not exist.
if not os.path.exists(filename):
    os.system('touch ' + filename)

# Read sections in ini file.
config = configparser.ConfigParser()
config.read(filename)

# Update section.
tag = 'mpi-' + str(1) + '-' + str(16384)
if tag not in config.sections(): config[tag] = {}
for ntask, elapsed_time in zip(ntasks, elapsed_times):
    config[tag][str(ntask)] = str(elapsed_time)

# Write section.
with open(filename, 'w') as f: config.write(f)

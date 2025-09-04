#!/usr/bin/env python

from subprocess import run
import time 
import configparser
import os
import shlex

nthreads = [1, 2, 4, 8, 16, 32, 64]
problem_sizes = [1024, 1280, 1600, 2048, 2560, 3264, 4096]
elapsed_times = []
mpi_prefix_args = ''
cmd_extra_args = ''
start_nthread_idx = 0
stop_nthread_idx = 7

for idx, nthread in enumerate(nthreads[start_nthread_idx:stop_nthread_idx]):
    # Build the cmd.
    problem_size_arg = [str(problem_sizes[idx])]
    cmd = shlex.split('srun') + shlex.split(f'-n 1') + shlex.split(mpi_prefix_args) + shlex.split('./main') + problem_size_arg + shlex.split(cmd_extra_args)

    # Set omp_num_threads. 
    os.environ['OMP_NUM_THREADS'] = str(f'{nthread}')

    # Run and time task
    start_time = time.perf_counter()
    print('Running: ', ' '.join(cmd), '.', flush=True)
    result = run(cmd)
    stop_time = time.perf_counter()

    
    # Calculate elapsed time.
    elapsed_time = stop_time - start_time
    print(f'Done in {elapsed_time} seconds.\n\n', flush=True)
    elapsed_times.append(elapsed_time)


filename = 'weak-scaling.ini'
# Create file if it does not exist.
if not os.path.exists(filename):
    os.system('touch ' + filename)

# Read sections in ini file.
config = configparser.ConfigParser()
config.read(filename)

# Update section.
tag = 'openmp-' + str(1) + '-' + str(1024)
if tag not in config.sections(): config[tag] = {}
for nthread, elapsed_time in zip(nthreads, elapsed_times):
    config[tag][str(nthread)] = str(elapsed_time)

# Write section.
with open(filename, 'w') as f: config.write(f)

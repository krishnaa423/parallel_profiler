#!/bin/bash

paraprof \
  --type strong-scaling \
  --mpi \
  --ntasks 1 2 4 8 16 32  \
  --nthreads 1 \
  --problem_sizes 16384 \
  --program 'python ./test.py' \
  --mpiexec srun
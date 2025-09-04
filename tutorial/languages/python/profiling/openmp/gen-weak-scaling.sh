#!/bin/bash

paraprof \
  --type weak-scaling \
  --openmp \
  --ntasks 1 \
  --nthreads 10 30 50 70 100 120 \
  --problem_sizes 8 \
  --program 'python ./test.py' \
  --mpiexec srun
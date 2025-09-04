#!/bin/bash

paraprof \
  --type weak-scaling \
  --mpi \
  --ntasks 1 2 3 4 5 6 7 8\
  --nthreads 1 \
  --problem_sizes 100000 \
  --program 'python ./test.py' \
  --mpiexec mpirun
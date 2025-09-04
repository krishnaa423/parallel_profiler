#!/bin/bash

paraprof \
  --type strong-scaling \
  --openmp \
  --ntasks 1 \
  --nthreads 1 2 3 4 5 6 7 8 \
  --problem_sizes 1000 \
  --program 'python ./test.py' \
  --mpiexec mpirun
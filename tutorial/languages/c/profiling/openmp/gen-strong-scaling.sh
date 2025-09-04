#!/bin/bash

paraprof \
  --type strong-scaling \
  --openmp \
  --ntasks 1 \
  --nthreads 10 30 50 70 100 120 \
  --problem_sizes 1000000000 \
  --program './main' \
  --mpiexec srun
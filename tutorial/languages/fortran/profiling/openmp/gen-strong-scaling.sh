#!/bin/bash

paraprof \
  --type strong-scaling \
  --openmp \
  --ntasks 1 \
  --nthreads 1 2 4 8 16 32 64 \
  --problem_sizes 1024 \
  --program './main' \
  --mpiexec srun
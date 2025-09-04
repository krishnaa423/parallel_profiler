#!/bin/bash

  # --problem_sizes 2048 2560 3264 4096 5184 6528 8192 \    # Takes about 40 seconds for each run. 
paraprof \
  --type weak-scaling \
  --openmp \
  --ntasks 1 \
  --nthreads 1 2 4 8 16 32 64 \
  --problem_sizes 1024 1280 1600 2048 2560 3264 4096 \
  --program './main' \
  --mpiexec srun

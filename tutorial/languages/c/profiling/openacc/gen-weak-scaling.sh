#!/bin/bash

paraprof \
  --type weak-scaling \
  --mpi \
  --ntasks 20 50 100 120 150 180 200 250 \
  --nthreads 1 \
  --problem_sizes 800000 \
  --program './main' \
  --mpiexec srun
#!/bin/bash

paraprof \
  --type strong-scaling \
  --mpi \
  --ntasks 20 50 100 120 150 180 200 250 \
  --nthreads 1 \
  --problem_sizes 1000000000 \
  --program './main' \
  --mpiexec srun
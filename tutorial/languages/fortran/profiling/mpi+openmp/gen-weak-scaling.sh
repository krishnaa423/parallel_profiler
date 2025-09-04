#!/bin/bash

paraprof \
  --type weak-scaling \
  --mpi \
  --ntasks 1 8 27 64 125 \
  --nthreads 1 \
  --problem_sizes 8192 16384 24576 32768 40960 \
  --program './main' \
  --mpiexec srun
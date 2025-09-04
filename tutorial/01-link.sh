#!/bin/bash

# OpenMP: code + scaling + debugging.
ln -sf ./languages/fortran/profiling/openmp ./02-openmp 

# OpenACC: code + debugging.
ln -sf ./languages/fortran/code_samples/openacc ./03-openacc 

# MPI: code + scaling + debugging.
ln -sf ./languages/python/profiling/mpi ./04-mpi-python 
ln -sf ./languages/fortran/profiling/mpi+openmp ./05-mpi-fortran 

# CUDA: code + debugging.
ln -sf ./languages/fortran/code_samples/cuda ./06-cuda 

# MPI+CUDA: code + debugging. 
ln -sf ./languages/fortran/code_samples/mpi+cuda ./07-mpi+cuda 
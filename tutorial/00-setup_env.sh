#!/bin/bash

conda deactivate

# Other exports.
export CRAY_ACCEL_TARGET=nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=2
export OMP_DEBUG=enabled        # For OpenMP debugging with TotalView.
# export ATP_ENABLED=1

module purge
# Load modules.
# Compilers and build tools.
module load PrgEnv-nvidia
module load cmake
# gpu stuff.
module load gpu
module load craype-accel-nvidia80
module load cudatoolkit
module load cudnn
module load nccl
# python and first principles code.
module load python
module load espresso
module load berkeleygw
# debuggers.
module load forge # The DDT debugger.
module load totalview   # Totalview debugger.
module load valgrind    # For profiling. 
module load gdb4hpc
module load cray-stat
# module load atp
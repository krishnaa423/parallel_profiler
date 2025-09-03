#!/bin/bash

# C. 
cd ../src/c/
mkdir build
cmake -S . -B build
cmake --build build
cd ../tutorial
ln -sf main_c ../src/c/build/main_c 

# Fortran.
cd ../src/fortran/
mkdir build
cmake -S . -B build
cmake --build build
cd ../tutorial
ln -sf main_fortran ../src/c/build/main_fortran

# Python with cython.
pip install -e ../
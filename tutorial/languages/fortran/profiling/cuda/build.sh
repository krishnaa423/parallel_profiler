#!/bin/bash

nvfortran -g -O0 -cuda -gpu=debug,lineinfo,ptxinfo -o main main.cuf
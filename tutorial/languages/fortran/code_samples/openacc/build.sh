#!/bin/bash

mpif90 -acc -O0 -g -gpu=debug,lineinfo,ptxinfo -Minfo=accel -o main main.f90
#!/bin/bash

mpif90 -g -O0 -acc -gpu=debug,lineinfo,ptxinfo -Minfo=accel -o main main.f90
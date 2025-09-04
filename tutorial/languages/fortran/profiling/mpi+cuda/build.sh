#!/bin/bash

mpif90 -g -O0 -cuda -gpu=debug,lineinfo,ptxinfo -o main main.cuf
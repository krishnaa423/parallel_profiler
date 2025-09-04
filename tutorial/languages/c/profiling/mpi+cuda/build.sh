#!/bin/bash

mpic++ -g -O0 -cuda -gpu=debug,lineinfo,ptxinfo -std=c++17 -o main main.cu
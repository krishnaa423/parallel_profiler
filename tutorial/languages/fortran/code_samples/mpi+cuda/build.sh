#!/bin/bash

mpic++ -std=c++17 -cuda -g -O0 -gpu=debug,lineinfo -o main_dbg main.c
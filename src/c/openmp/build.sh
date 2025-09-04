#!/bin/bash

CC=gcc-11
CHUNK_SIZE=4
ARRAY_SIZE=100
ARRAY_INIT_VALUE=2.0

rm -rf ./build
mkdir -p ./build

cmake \
  -DCMAKE_C_COMPILER=${CC} \
  -DCHUNK_SIZE=${CHUNK_SIZE} \
  -DARRAY_SIZE=${ARRAY_SIZE} \
  -DARRAY_INIT_VALUE="${ARRAY_INIT_VALUE}" \
  -S . -B build

cmake --build build --verbose
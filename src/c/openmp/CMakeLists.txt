cmake_minimum_required(VERSION 3.15)
project(array_pow10 C)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g -fopenmp -DARRAY_INIT_VALUE=${ARRAY_INIT_VALUE} -DARRAY_SIZE=${ARRAY_SIZE} -DCHUNK_SIZE=${CHUNK_SIZE}")

add_library(openmp_pow10 STATIC array_pow10.c)
add_executable(test_openmp_pow10_c main.c)
target_link_libraries(test_openmp_pow10_c openmp_pow10 gomp)

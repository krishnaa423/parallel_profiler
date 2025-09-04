#include <stdio.h>
#include <omp.h>

void array_pow10(double *a, int n) {
    for (long t = 0; t < 10; ++t) {
        #pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
        for (long i = 0; i < n; ++i) {
            a[i] = a[i] * a[i];
        }
    }
}

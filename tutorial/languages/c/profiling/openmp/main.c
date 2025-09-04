// Compile:  gcc -O3 -fopenmp -o omp_dot omp_dot.c
// Run:      ./omp_dot 100000000
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s n\n", argv[0]); return 1; }
    long long n = atoll(argv[1]);
    if (n <= 0) { fprintf(stderr, "n must be > 0\n"); return 1; }

    double t0 = now_sec();

    double *A = (double*)malloc(sizeof(double)*n);
    double *B = (double*)malloc(sizeof(double)*n);
    if (!A || !B) { fprintf(stderr, "allocation failed\n"); return 2; }

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) {
        A[i] = (double)i;
        B[i] = 1.0 / (double)(i + 1);
    }

    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (long long i = 0; i < n; ++i) {
        sum += A[i] * B[i];
    }

    double t1 = now_sec();
    printf("[OpenMP] n=%lld threads=%d  sum=%.12f  time=%.3f s\n",
           n, omp_get_max_threads(), sum, t1 - t0);

    free(A); free(B);
    return 0;
}

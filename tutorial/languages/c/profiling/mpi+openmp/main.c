// Compile:  mpicc -O3 -fopenmp -o mpi_omp_dot mpi_omp_dot.c
// Run:      mpirun -np 4 ./mpi_omp_dot 100000000
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) fprintf(stderr, "usage: %s n\n", argv[0]);
        MPI_Finalize(); return 1;
    }
    long long n = atoll(argv[1]);
    if (n <= 0) { if (rank==0) fprintf(stderr, "n must be > 0\n"); MPI_Finalize(); return 1; }

    long long chunk = n / size;
    long long r = n % size;
    long long local_n = chunk + (rank < r ? 1 : 0);
    long long start = rank * chunk + (rank < r ? rank : r);
    long long end = start + local_n; // [start, end)

    double t0 = now_sec();

    double *A = (double*)malloc(sizeof(double)*local_n);
    double *B = (double*)malloc(sizeof(double)*local_n);
    if (!A || !B) { fprintf(stderr, "rank %d: allocation failed\n", rank); MPI_Abort(MPI_COMM_WORLD, 2); }

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < local_n; ++i) {
        long long g = start + i;
        A[i] = (double)g;
        B[i] = 1.0 / (double)(g + 1);
    }

    double local_sum = 0.0;
    #pragma omp parallel for reduction(+:local_sum) schedule(static)
    for (long long i = 0; i < local_n; ++i) {
        local_sum += A[i] * B[i];
    }

    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double t1 = now_sec();
    if (rank == 0) {
        printf("[MPI+OpenMP] n=%lld ranks=%d threads/rank=%d  sum=%.12f  time=%.3f s\n",
               n, size, omp_get_max_threads(), global_sum, t1 - t0);
    }

    free(A); free(B);
    MPI_Finalize();
    return 0;
}

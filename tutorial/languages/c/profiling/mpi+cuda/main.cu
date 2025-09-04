// Recommended compile (works on many systems):
//   nvcc -O3 -arch=sm_60 -ccbin mpicxx -o mpi_cuda_dot mpi_cuda_dot.cu
// Alternatives (cluster-dependent):
//   mpicxx -O3 mpi_cuda_dot.cu -lcudart -L$CUDA_HOME/lib64 -o mpi_cuda_dot
// Run: mpirun -np 4 ./mpi_cuda_dot 100000000

#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <cuda_runtime.h>

__global__ void dot_kernel(const double* __restrict__ A,
                           const double* __restrict__ B,
                           double* sum, long long n, long long start_idx) {
    double local = 0.0;
    long long gid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long k = gid; k < n; k += stride) {
        long long g = start_idx + k;
        double a = (double)g;
        double b = 1.0 / (double)(g + 1);
        local += a * b;
    }
    atomicAdd(sum, local);
}

static inline void check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank=0, size=1; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank==0) fprintf(stderr, "usage: %s n\n", argv[0]);
        MPI_Finalize(); return 1;
    }
    long long n = atoll(argv[1]);
    if (n <= 0) { if (rank==0) fprintf(stderr, "n must be > 0\n"); MPI_Finalize(); return 1; }

    // Decompose global range
    long long chunk = n / size;
    long long r = n % size;
    long long local_n = chunk + (rank < r ? 1 : 0);
    long long start = rank * chunk + (rank < r ? rank : r);

    // Pick a GPU per rank (round-robin)
    int devCount = 0; cudaGetDeviceCount(&devCount);
    if (devCount == 0) { if (rank==0) fprintf(stderr, "No CUDA devices found\n"); MPI_Abort(MPI_COMM_WORLD, 3); }
    int dev = rank % devCount;
    check(cudaSetDevice(dev), "set device");

    // Device accumulator
    double *dSum = nullptr;
    check(cudaMalloc(&dSum, sizeof(double)), "cudaMalloc dSum");
    check(cudaMemset(dSum, 0, sizeof(double)), "memset dSum");

    // Time
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    // Launch: compute A[i], B[i] on the fly on device to avoid H2D copies
    int tpb = 256;
    int maxBlocks = 1024;
    int blocks = (int)((local_n + tpb - 1) / tpb);
    if (blocks > maxBlocks) blocks = maxBlocks;

    dot_kernel<<<blocks, tpb>>>(nullptr, nullptr, dSum, local_n, start);
    check(cudaGetLastError(), "kernel");
    check(cudaDeviceSynchronize(), "sync");

    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0.0f; cudaEventElapsedTime(&ms, t0, t1);

    double local_sum = 0.0, global_sum = 0.0;
    check(cudaMemcpy(&local_sum, dSum, sizeof(double), cudaMemcpyDeviceToHost), "D2H sum");
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("[MPI+CUDA] n=%lld ranks=%d (gpu/devCount may vary)  sum=%.12f  time=%.3f s\n",
               n, size, global_sum, ms*1e-3);
    }

    cudaFree(dSum);
    MPI_Finalize();
    return 0;
}

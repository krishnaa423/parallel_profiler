// Compile: nvcc -O3 -arch=sm_60 -o cuda_dot cuda_dot.cu
// Run:     ./cuda_dot 100000000
// Note: uses atomicAdd on double (needs sm_60+). Change to float if needed.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void dot_kernel(const double* __restrict__ A,
                           const double* __restrict__ B,
                           double* sum, long long n) {
    double local = 0.0;
    long long gid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = gid; i < n; i += stride) {
        local += A[i] * B[i];
    }
    atomicAdd(sum, local);
}

static inline void check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e)); exit(2); }
}

static inline double now_sec() {
    cudaDeviceSynchronize();
    static cudaEvent_t start, stop;
    static bool inited = false;
    if (!inited) { cudaEventCreate(&start); cudaEventCreate(&stop); inited = true; }
    cudaEventRecord(start, 0);
    cudaEventRecord(stop, 0); // just to create valid events
    return 0.0; // not used
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s n\n", argv[0]); return 1; }
    long long n = atoll(argv[1]);
    if (n <= 0) { fprintf(stderr, "n must be > 0\n"); return 1; }

    double* hA = (double*)malloc(sizeof(double)*n);
    double* hB = (double*)malloc(sizeof(double)*n);
    if (!hA || !hB) { fprintf(stderr, "host allocation failed\n"); return 2; }
    for (long long i = 0; i < n; ++i) { hA[i] = (double)i; hB[i] = 1.0/(double)(i+1); }

    double *dA = nullptr, *dB = nullptr, *dSum = nullptr;
    check(cudaMalloc(&dA, n*sizeof(double)), "cudaMalloc dA");
    check(cudaMalloc(&dB, n*sizeof(double)), "cudaMalloc dB");
    check(cudaMalloc(&dSum, sizeof(double)), "cudaMalloc dSum");
    check(cudaMemcpy(dA, hA, n*sizeof(double), cudaMemcpyHostToDevice), "H2D A");
    check(cudaMemcpy(dB, hB, n*sizeof(double), cudaMemcpyHostToDevice), "H2D B");
    check(cudaMemset(dSum, 0, sizeof(double)), "Memset sum");

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    int tpb = 256;
    int maxBlocks = 1024;
    int blocks = (int)((n + tpb - 1) / tpb);
    if (blocks > maxBlocks) blocks = maxBlocks;

    dot_kernel<<<blocks, tpb>>>(dA, dB, dSum, n);
    check(cudaGetLastError(), "kernel launch");
    check(cudaDeviceSynchronize(), "sync");

    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0.0f; cudaEventElapsedTime(&ms, t0, t1);

    double sum = 0.0;
    check(cudaMemcpy(&sum, dSum, sizeof(double), cudaMemcpyDeviceToHost), "D2H sum");

    printf("[CUDA] n=%lld  sum=%.12f  time=%.3f s\n", n, sum, ms*1e-3);

    cudaFree(dA); cudaFree(dB); cudaFree(dSum);
    free(hA); free(hB);
    return 0;
}

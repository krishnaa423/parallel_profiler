// matmul_cuda_c.c
// Build (NVIDIA HPC SDK):
//   nvc -cuda -O3 matmul_cuda_c.c -o main
//
// Run:
//   ./main [N]

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifndef TILE
#define TILE 32
#endif

#define CUDA_OK(call) do {                                     \
  cudaError_t _err = (call);                                   \
  if (_err != cudaSuccess) {                                   \
    fprintf(stderr, "CUDA error %s at %s:%d\n",                \
            cudaGetErrorString(_err), __FILE__, __LINE__);     \
    exit(1);                                                   \
  }                                                            \
} while (0)

__global__ void matmul_tiled(const double* __restrict__ A,
                             const double* __restrict__ B,
                             double* __restrict__ C,
                             int N)
{
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * TILE + ty;  // 0-based in CUDA C
    int col = bx * TILE + tx;

    double sum = 0.0;

    // number of tiles along the K dimension
    int numTiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;

        // Guarded loads into shared memory
        if (row < N && aCol < N) {
            As[ty][tx] = A[row * N + aCol];
        } else {
            As[ty][tx] = 0.0;
        }

        if (bRow < N && col < N) {
            Bs[ty][tx] = B[bRow * N + col];
        } else {
            Bs[ty][tx] = 0.0;
        }

        __syncthreads();

        // Multiply-accumulate within the tile
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main(int argc, char** argv)
{
    int N = 512; // default
    if (argc >= 2) {
        long v = strtol(argv[1], NULL, 10);
        if (v <= 0) {
            fprintf(stderr, "Usage: %s [N]  (N must be a positive integer)\n", argv[0]);
            return 3;
        }
        N = (int)v;
    }

    printf("Using N = %d\n", N);

    size_t bytes = (size_t)N * (size_t)N * sizeof(double);

    // Host allocations
    double *A_h = (double*)malloc(bytes);
    double *B_h = (double*)malloc(bytes);
    double *C_h = (double*)malloc(bytes);
    if (!A_h || !B_h || !C_h) {
        fprintf(stderr, "Host allocation failed (try a smaller N).\n");
        return 4;
    }

    // Initialize with 1-based i,j semantics to match the Fortran code:
    // A(i,j) = i + j, B(i,j) = i - j, for i,j in 1..N
    // Our arrays are 0-based, so use (i+1), (j+1)
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            A_h[i + (size_t)j * N] = (double)((i + 1) + (j + 1));
            B_h[i + (size_t)j * N] = (double)((i + 1) - (j + 1));
        }
    }

    // Device allocations
    double *A_d = NULL, *B_d = NULL, *C_d = NULL;
    CUDA_OK(cudaMalloc((void**)&A_d, bytes));
    CUDA_OK(cudaMalloc((void**)&B_d, bytes));
    CUDA_OK(cudaMalloc((void**)&C_d, bytes));

    // H2D copies
    CUDA_OK(cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice));

    // Launch configuration
    dim3 block(TILE, TILE, 1);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE, 1);

    matmul_tiled<<<grid, block>>>(A_d, B_d, C_d, N);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    // D2H copy
    CUDA_OK(cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost));

    // Print two elements analogous to Fortran's C(1,1) and C(n,n)
    // In 0-based: C[0,0] and C[N-1,N-1]
    printf("C(1,1)=%.6f  C(n,n)=%.6f\n",
           C_h[0],
           C_h[(size_t)(N - 1) * N + (N - 1)]);

    // Cleanup
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}

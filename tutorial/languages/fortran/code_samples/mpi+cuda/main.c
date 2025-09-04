// Compile with: mpic++ -std=c++17 -cuda -g -O0 -gpu=debug,lineinfo -o main_dbg main.c
// Run with:     mpirun -np 4 ./main

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_MPI(call) do {                                     \
  int _e = (call);                                               \
  if (_e != MPI_SUCCESS) {                                       \
    fprintf(stderr, "MPI error at %s:%d\n", __FILE__, __LINE__); \
    MPI_Abort(MPI_COMM_WORLD, _e);                               \
  }                                                              \
} while(0)

#define CHECK_CUDA(call) do {                                       \
  cudaError_t _e = (call);                                          \
  if (_e != cudaSuccess) {                                          \
    fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
            __FILE__, __LINE__, cudaGetErrorString(_e));            \
    MPI_Abort(MPI_COMM_WORLD, -1);                                  \
  }                                                                 \
} while(0)

__global__ void vadd(const float* a, const float* b, float* c, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
  CHECK_MPI(MPI_Init(&argc, &argv));

  int world_rank = 0, world_size = 1;
  CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
  CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

  // Determine local rank (ranks that share a node) to map GPUs per node
  MPI_Comm local_comm;
  CHECK_MPI(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm));
  int local_rank = 0;
  CHECK_MPI(MPI_Comm_rank(local_comm, &local_rank));
  MPI_Comm_free(&local_comm);

  int num_devices = 0;
  CHECK_CUDA(cudaGetDeviceCount(&num_devices));
  if (num_devices == 0) {
    if (world_rank == 0) fprintf(stderr, "No CUDA devices found.\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  int dev = local_rank % num_devices;
  CHECK_CUDA(cudaSetDevice(dev));

  // Total problem size (can pass as argv[1]); default: 1<<22 elements
  size_t N_total = (argc > 1) ? strtoull(argv[1], NULL, 10) : (1ULL << 22);

  // Partition the work as evenly as possible
  size_t base = N_total / world_size;
  size_t rem  = N_total % world_size;
  size_t n_local = base + (world_rank < rem ? 1 : 0);

  // Compute start offset for this rank
  size_t offset = world_rank * base + (size_t)(world_rank < rem ? world_rank : rem);

  // Allocate and initialize host input for local chunk
  float *h_a = (float*)malloc(n_local * sizeof(float));
  float *h_b = (float*)malloc(n_local * sizeof(float));
  float *h_c = (float*)malloc(n_local * sizeof(float));
  if (!h_a || !h_b || !h_c) {
    fprintf(stderr, "Host allocation failed at rank %d\n", world_rank);
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  // Simple deterministic init: a[i]=i, b[i]=2*i (using global index)
  for (size_t i = 0; i < n_local; ++i) {
    size_t gi = offset + i;
    h_a[i] = (float)gi;
    h_b[i] = 2.0f * (float)gi;
  }

  // Allocate device memory
  float *d_a = NULL, *d_b = NULL, *d_c = NULL;
  CHECK_CUDA(cudaMalloc((void**)&d_a, n_local * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&d_b, n_local * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&d_c, n_local * sizeof(float)));

  // Copy to device
  CHECK_CUDA(cudaMemcpy(d_a, h_a, n_local * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b, n_local * sizeof(float), cudaMemcpyHostToDevice));

  // Launch kernel
  const int TPB = 256;
  int blocks = (int)((n_local + TPB - 1) / TPB);
  vadd<<<blocks, TPB>>>(d_a, d_b, d_c, n_local);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // Copy result back
  CHECK_CUDA(cudaMemcpy(h_c, d_c, n_local * sizeof(float), cudaMemcpyDeviceToHost));

  // Gather counts/displacements to root
  size_t *recv_counts = NULL, *displs = NULL;
  float *h_c_global = NULL;

  if (world_rank == 0) {
    recv_counts = (size_t*)malloc(world_size * sizeof(size_t));
    displs      = (size_t*)malloc(world_size * sizeof(size_t));
  }

  // Gather sizes
  CHECK_MPI(MPI_Gather(&n_local, 1, MPI_UNSIGNED_LONG_LONG,
                       recv_counts, 1, MPI_UNSIGNED_LONG_LONG,
                       0, MPI_COMM_WORLD));

  if (world_rank == 0) {
    // Compute displacements in elements
    size_t running = 0;
    for (int r = 0; r < world_size; ++r) {
      displs[r] = running;
      running  += recv_counts[r];
    }
    h_c_global = (float*)malloc(N_total * sizeof(float));
    if (!h_c_global) {
      fprintf(stderr, "Root host allocation failed\n");
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }

  // Create a datatype for counts in floats
  // Use MPI_Gatherv to gather variable-sized chunks
  CHECK_MPI(MPI_Gatherv(h_c, (int)n_local, MPI_FLOAT,
                        h_c_global,
                        (int*)recv_counts, (int*)displs, MPI_FLOAT,
                        0, MPI_COMM_WORLD));

  // Quick correctness check on root for a few samples
  if (world_rank == 0) {
    int ok = 1;
    for (size_t i = 0; i < N_total; i += (N_total / 7 + 1)) {
      float expected = (float)i + 2.0f * (float)i; // 3*i
      if (fabsf(h_c_global[i] - expected) > 1e-3f) { ok = 0; break; }
    }
    printf("vadd %s (world_size=%d)\n", ok ? "OK" : "FAILED", world_size);
  }

  // Cleanup
  free(h_a); free(h_b); free(h_c);
  if (world_rank == 0) {
    free(h_c_global); free(recv_counts); free(displs);
  }
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));

  CHECK_MPI(MPI_Finalize());
  return 0;
}

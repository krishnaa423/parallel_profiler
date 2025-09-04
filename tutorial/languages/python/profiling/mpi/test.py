# MPI matrix-matrix multiply with ring-panel algorithm
# Usage: mpirun -np P python test.py N

import os
import sys, math, time
import numpy as np
from mpi4py import MPI
# # Make sure BLAS doesn't spawn threads just to track MPI scaling only.
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")
# os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

def split_sizes(n, p):
    base = n // p
    r = n % p
    # first r ranks get (base+1), the rest get base
    counts = [base + 1 if i < r else base for i in range(p)]
    offs = [0]*p
    for i in range(1, p):
        offs[i] = offs[i-1] + counts[i-1]
    return counts, offs

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if len(sys.argv) != 2:
        if rank == 0:
            print("Usage: mpirun -np P python mm_ring.py N", file=sys.stderr)
        sys.exit(1)

    try:
        N = int(sys.argv[1])
        assert N >= 1
    except Exception:
        if rank == 0:
            print("N must be a positive integer", file=sys.stderr)
        sys.exit(1)

    dtype = np.float64

    # Row-block distribution for A (and C) and row-block (k-panels) for B
    row_counts, row_offs = split_sizes(N, size)    # rows owned by each rank
    k_counts,   k_offs   = row_counts, row_offs    # reuse for B's row panels

    m_i = row_counts[rank]   # local rows of A and C
    k_i = k_counts[rank]     # local B panel rows

    # Allocate local blocks
    A_local = np.ones((m_i, N), dtype=dtype)       # A_i = 1 for easy verification
    B_curr  = np.ones((k_i, N), dtype=dtype)       # current B panel held by this rank
    C_local = np.zeros((m_i, N), dtype=dtype)

    # Ring neighbors
    left  = (rank - 1) % size
    right = (rank + 1) % size

    # Each process starts with its own panel (owner = rank)
    owner = rank

    comm.Barrier()
    t0 = MPI.Wtime()

    # Ring over all B panels
    for step in range(size):
        # Multiply with the columns of A that correspond to the current panel
        ks = k_offs[owner]
        ke = ks + k_counts[owner]
        if m_i > 0 and k_counts[owner] > 0:
            # A_sub: (m_i x k_blk), B_curr: (k_blk x N)
            A_sub = A_local[:, ks:ke]
            # Use NumPy/BLAS for local GEMM
            C_local += A_sub @ B_curr

        # Rotate B panels (skip comms if size==1)
        if size > 1:
            next_owner = (owner - 1) % size
            recv_rows = k_counts[next_owner]
            # Prepare receive buffer with the correct shape
            B_next = np.empty((recv_rows, N), dtype=dtype)
            # Send current panel to the right, receive new panel from the left
            comm.Sendrecv(sendbuf=B_curr, dest=right, sendtag=step,
                          recvbuf=B_next,  source=left,  recvtag=step)
            B_curr = B_next
            owner = next_owner

    comm.Barrier()
    t1 = MPI.Wtime()
    dt = t1 - t0

    # Verification (C should be N everywhere because A=B=1)
    if C_local.size > 0:
        local_err = float(np.max(np.abs(C_local - N)))
    else:
        local_err = 0.0
    max_err = comm.allreduce(local_err, op=MPI.MAX)

    # Report the slowest rank time (what you want for scaling graphs)
    t_max = comm.allreduce(dt, op=MPI.MAX)

    # Total flops for dense GEMM: 2*N^3  (counting mul+add)
    gflops = (2.0 * N * N * N) / (t_max * 1e9) if t_max > 0 else float('inf')

    if rank == 0:
        # machine-parsable one-liner for plotting later
        print(f"RESULT algo=ring_mm N={N} P={size} time={t_max:.6f}s gflops={gflops:.3f} max_err={max_err:.3e}")

if __name__ == "__main__":
    main()

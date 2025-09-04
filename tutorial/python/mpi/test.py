#region modules
from mpi4py import MPI
import sys
import numpy as np
#endregion

#region variables
#endregion

#region functions
def axpy_dot_reduce(n: int):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- Block distribution of global size n ---
    base = n // size
    rem = n % size
    local_n = base + (1 if rank < rem else 0)
    start = rank * base + min(rank, rem)
    stop = start + local_n

    # --- Local arrays ---
    x = np.ones(local_n, dtype=np.float64)
    y = np.full(local_n, rank + 1, dtype=np.float64)

    # --- One AXPY ---
    y = 2.0 * x + y

    # --- Local dot and global reduction ---
    local_dot = np.dot(x, y)
    global_dot = comm.allreduce(local_dot, op=MPI.SUM)

    if rank == 0:
        print(f"Global size = {n}, Final dot product = {global_dot:.6e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <matrix_size>")
        sys.exit(1)

    n = int(sys.argv[1])
    axpy_dot_reduce(n)
#endregion

#region classes
#endregion

#region main
if __name__=='__main__':
    main()
#endregion
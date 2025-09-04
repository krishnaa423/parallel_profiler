#region modules
from mpi4py import MPI
import sys
import cupy as cp 
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
    rem  = n % size
    local_n = base + (1 if rank < rem else 0)
    # start/stop are not used further but shown for clarity
    start = rank * base + min(rank, rem)
    stop  = start + local_n

    # --- Device arrays ---
    x = cp.ones(local_n, dtype=cp.float64)
    y = cp.full(local_n, rank + 1, dtype=cp.float64)

    # --- One AXPY on device: y = 2*x + y ---
    y = 2.0 * x + y

    # --- Local dot on device ---
    if local_n > 0:
        local_dot_dev = cp.array([cp.dot(x, y)], dtype=cp.float64)  # shape (1,)
    else:
        local_dot_dev = cp.array([0.0], dtype=cp.float64)

    # Ensure compute is ready before MPI touches device buffers
    cp.cuda.get_current_stream().synchronize()

    # --- CUDA-aware Allreduce directly on device buffers ---
    global_dot_dev = cp.empty_like(local_dot_dev)
    comm.Allreduce([local_dot_dev, MPI.DOUBLE],
                   [global_dot_dev, MPI.DOUBLE],
                   op=MPI.SUM)
    
    if rank == 0:
        global_dot = float(global_dot_dev[0].get())  # copy one scalar to host for printing
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
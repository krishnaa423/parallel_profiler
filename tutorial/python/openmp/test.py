#region modules
import numpy as np
import sys
#endregion

#region variables
#endregion

#region functions
def matmult(n: int):
    A = np.ones((n, n))*1.0
    B = np.ones((n, n))*3.0
    C = A @ B   # triggers BLAS/OpenMP
    return C

def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <matrix_size>")
        sys.exit(1)

    n = int(sys.argv[1])
    matmult(n)
#endregion

#region classes
#endregion

#region main
if __name__=='__main__':
    main()
#endregion
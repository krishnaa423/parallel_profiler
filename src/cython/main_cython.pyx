#region modules
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, language_level=3
import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange
cimport cython
#endregion

#region variables
#endregion

#region functions
cdef inline double _dot_row(double[:, ::1] A, double[::1] x, Py_ssize_t i) nogil:
    cdef Py_ssize_t j, m = A.shape[1]
    cdef double s = 0.0
    for j in range(m):
        s += A[i, j] * x[j]
    return s

cpdef gemv_cython(double[:, ::1] A, double[::1] x):
    cdef Py_ssize_t n = A.shape[0]

    # allocate under GIL
    cdef cnp.ndarray[cnp.float64_t, ndim=1] b = np.empty(n, dtype=np.float64)
    cdef double[::1] bv = b  # memoryview for nogil writes
    cdef Py_ssize_t i

    for i in prange(n, nogil=True, schedule='static'):
        bv[i] = _dot_row(A, x, i)  # all C inside

    return b
#endregion

#region classes
#endregion
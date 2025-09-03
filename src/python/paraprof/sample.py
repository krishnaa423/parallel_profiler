#region modules
import paraprof.cython_ext as cx
import numpy as np
#endregion

#region variables
#endregion

#region functions
def gemv(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    b = cx.gemv_cython(A, x)
    return b
#endregion

#region classes
#endregion
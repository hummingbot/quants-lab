cimport cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double sma_cython(np.ndarray[np.double_t, ndim=1] data, int window):
    cdef Py_ssize_t i, n = data.shape[0]
    cdef double s = 0.0
    if n < window:
        window = n
    for i in range(window):
        s += data[i]
    return s / window

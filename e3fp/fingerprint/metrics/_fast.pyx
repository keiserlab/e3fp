#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#
# Speed up metrics for comparisons that are very slow with NumPy methods.
#
# Author: Seth Axen
# E-mail: seth.axen@gmail.com

import numpy as np
cimport numpy as np
ctypedef np.float64_t cDOUBLE


cpdef np.ndarray[cDOUBLE, ndim=2] fast_soergel(X, Y, bint sparse=False):
    cdef np.ndarray[cDOUBLE, ndim=2] S = np.empty(
        (X.shape[0], Y.shape[0]), dtype=np.float64)
    if sparse:
        _sparse_soergel(X.data, X.indices, X.indptr,
                        Y.data, Y.indices, Y.indptr, S)
    else:
        _dense_soergel(X, Y, S)
    return S


cdef void _dense_soergel(cDOUBLE[:, ::1] X,
                         cDOUBLE[:, ::1] Y,
                         cDOUBLE[:, ::1] S):
    cdef:
        np.npy_intp ix, iy, j
        cDOUBLE sum_abs_diff, sum_max, diff
    with nogil:
        for ix in range(S.shape[0]):
            for iy in range(S.shape[1]):
                sum_abs_diff = 0
                sum_max = 0
                for j in range(X.shape[1]):
                    diff = X[ix, j] - Y[iy, j]
                    if diff > 0:
                        sum_abs_diff += diff
                        sum_max += X[ix, j]
                    else:
                        sum_abs_diff -= diff
                        sum_max += Y[iy, j]

                if sum_max == 0:
                    S[ix, iy] = 0
                    continue
                S[ix, iy] = 1 - sum_abs_diff / sum_max


cdef void _sparse_soergel(cDOUBLE[::1] Xdata,
                          int[::1] Xindices,
                          int[::1] Xindptr,
                          cDOUBLE[::1] Ydata,
                          int[::1] Yindices,
                          int[::1] Yindptr,
                          cDOUBLE[:, ::1] S):
    cdef:
        np.npy_intp ix, iy, jx, jy, jxindmax, jyindmax, jxind, jyind
        cDOUBLE sum_abs_diff, sum_max, diff
    with nogil:
        for ix in range(S.shape[0]):
            if Xindptr[ix] == Xindptr[ix + 1]:
                for iy in range(S.shape[1]):  # no X values in row
                    S[ix, iy] = 0
                continue
            jxindmax = Xindptr[ix + 1] - 1
            for iy in range(S.shape[1]):
                if Yindptr[iy] == Yindptr[iy + 1]:  # no Y values in row
                    S[ix, iy] = 0
                    continue

                sum_abs_diff = 0
                sum_max = 0
                # Implementation of the final step of merge sort
                jyindmax = Yindptr[iy + 1] - 1
                jx = Xindptr[ix]
                jy = Yindptr[iy]
                while jx <= jxindmax and jy <= jyindmax:
                    jxind = Xindices[jx]
                    jyind = Yindices[jy]   
                    if jxind < jyind:
                        sum_max += Xdata[jx]
                        sum_abs_diff += Xdata[jx]
                        jx += 1
                    elif jyind < jxind:
                        sum_max += Ydata[jy]
                        sum_abs_diff += Ydata[jy]
                        jy += 1
                    else:
                        diff = Xdata[jx] - Ydata[jy]
                        if diff > 0:
                            sum_abs_diff += diff
                            sum_max += Xdata[jx]
                        else:
                            sum_abs_diff -= diff
                            sum_max += Ydata[jy]
                        jx += 1
                        jy += 1

                while jx <= jxindmax:
                    sum_max += Xdata[jx]
                    sum_abs_diff += Xdata[jx]
                    jx += 1

                while jy <= jyindmax:
                    sum_max += Ydata[jy]
                    sum_abs_diff += Ydata[jy]
                    jy += 1

                if sum_max == 0:
                    S[ix, iy] = 0
                    continue
                S[ix, iy] = 1 - sum_abs_diff / sum_max

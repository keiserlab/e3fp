"""Fingerprint array comparison metrics.

Each is fully compatible with both dense and sparse inputs.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
from __future__ import division

import numpy as np
import scipy
from scipy.sparse import csr_matrix, issparse, vstack
from e3fp.util import maybe_jit


def tanimoto(X, Y=None):
    """Compute the Tanimoto coefficients between `X` and `Y`.

    Data must be binary. This is not checked.

    Parameters
    ----------
    X : array_like or sparse matrix
        with shape (`n_fprints_X`, `n_bits`).
    Y : array_like or sparse matrix, optional
        with shape (`n_fprints_Y`, `n_bits`).

    Returns
    -------
    tanimoto : array of shape (`n_fprints_X`, `n_fprints_Y`)

    See Also
    --------
    soergel: Analog to Tanimoto for non-binary data.
    cosine, dice, pearson
    """
    X, Y = _check_array_pair(X, Y)
    Xbits, Ybits, XYbits = _get_bitcount_arrays(X, Y, return_XYbits=True)
    with np.errstate(divide="ignore"):  # handle 0 in denominator
        return np.asarray(np.nan_to_num(XYbits / (Xbits + Ybits.T - XYbits)))


def soergel(X, Y=None):
    """Compute the Soergel similarities between `X` and `Y`.

    Soergel similarity is the complement of Soergel distance and can be
    thought of as the analog of the Tanimoto coefficient for count/float-based
    data. For binary data, it is equivalent to the Tanimoto coefficient.

    Parameters
    ----------
    X : array_like or sparse matrix
        with shape (`n_fprints_X`, `n_bits`).
    Y : array_like or sparse matrix, optional
        with shape (`n_fprints_Y`, `n_bits`).

    Returns
    -------
    soergel : array of shape (`n_fprints_X`, `n_fprints_Y`)

    Notes
    --------
    If Numba is available, this function is jit-compiled and much more efficient.

    See Also
    --------
    tanimoto: A fast version of this function for binary data.
    pearson: Pearson correlation, also appropriate for non-binary data.
    cosine, dice
    """
    X, Y = _check_array_pair(X, Y)
    S = np.empty((X.shape[0], Y.shape[0]), dtype=np.float64)
    if issparse(X):
        return _sparse_soergel(X.data, X.indices, X.indptr,
                               Y.data, Y.indices, Y.indptr, S)
    return _dense_soergel(X, Y, S)

def dice(X, Y=None):
    """Compute the Dice coefficients between `X` and `Y`.

    Data must be binary. This is not checked.

    Parameters
    ----------
    X : array_like or sparse matrix
        with shape (`n_fprints_X`, `n_bits`).
    Y : array_like or sparse matrix, optional
        with shape (`n_fprints_Y`, `n_bits`).

    Returns
    -------
    dice : array of shape (`n_fprints_X`, `n_fprints_Y`)

    See Also
    --------
    cosine, soergel, tanimoto, pearson
    """
    X, Y = _check_array_pair(X, Y)
    Xbits, Ybits, XYbits = _get_bitcount_arrays(X, Y, return_XYbits=True)
    with np.errstate(divide="ignore"):  # handle 0 in denominator
        return np.asarray(np.nan_to_num(2 * XYbits / (Xbits + Ybits.T)))


def cosine(X, Y=None, assume_binary=False):
    """Compute the Cosine similarities between `X` and `Y`.

    Parameters
    ----------
    X : array_like or sparse matrix
        with shape (`n_fprints_X`, `n_bits`).
    Y : array_like or sparse matrix, optional
        with shape (`n_fprints_Y`, `n_bits`).
    assume_binary : bool, optional
        Assume data is binary (results in efficiency boost). If data is not
        binary, the result will be incorrect.

    Returns
    -------
    cosine : array of shape (`n_fprints_X`, `n_fprints_Y`)

    See Also
    --------
    dice, soergel, tanimoto
    """
    X, Y = _check_array_pair(X, Y)
    if not issparse(X):
        return 1.0 - scipy.spatial.distance.cdist(X, Y, metric="cosine")
    if assume_binary:
        Xbits, Ybits, XYbits = _get_bitcount_arrays(X, Y, return_XYbits=True)
        with np.errstate(divide="ignore"):  # handle 0 in denominator
            return np.asarray(np.nan_to_num(XYbits / np.sqrt(Xbits * Ybits.T)))
    else:
        return _sparse_cosine(X, Y)


def pearson(X, Y=None):
    """Compute the Pearson correlation between `X` and `Y`.

    Parameters
    ----------
    X : array_like or sparse matrix
        with shape (`n_fprints_X`, `n_bits`).
    Y : array_like or sparse matrix, optional
        with shape (`n_fprints_Y`, `n_bits`).

    Returns
    -------
    pearson : array of shape (`n_fprints_X`, `n_fprints_Y`)


    See Also
    --------
    soergel: Soergel similarity for non-binary data
    cosine, dice, tanimoto
    """
    X, Y = _check_array_pair(X, Y)
    Xlen = X.shape[0]
    if issparse(X):
        X = vstack((X, Y), format="csr")
        X = X - X.mean(axis=1)
        cov = (X * X.T) / (X.shape[1] - 1.0)
        d = np.sqrt(np.diag(cov))
        with np.errstate(divide="ignore"):  # handle 0 in denominator
            pearson = cov / np.outer(d, d)
    else:
        with np.errstate(divide="ignore"):  # handle 0 in denominator
            pearson = scipy.corrcoef(X, Y)
    return np.asarray(np.nan_to_num(pearson[:Xlen, Xlen:]))


def _check_array(arr, dtype=np.double, force_sparse=False):
    if force_sparse or issparse(arr):
        return csr_matrix(arr, copy=False, dtype=dtype)
    else:
        return arr.astype(dtype, copy=False)


def _check_array_pair(X, Y=None, dtype=np.double, force_sparse=False):
    if Y is not None and X.shape[1] != Y.shape[1]:
        raise ValueError("Arrays must have same width.")
    if force_sparse or issparse(X) or issparse(Y):
        force_sparse = True  # ensure if one is sparse, all are sparse.
    X = _check_array(X, dtype=dtype, force_sparse=force_sparse)
    if Y is None or Y is X:
        Y = X
    else:
        Y = _check_array(Y, dtype=dtype, force_sparse=force_sparse)
    return X, Y


def _get_bitcount_arrays(X, Y, return_XYbits=False):
    if issparse(X):
        Xbits = np.sum(X, axis=1)
        if Y is X:
            Ybits = Xbits
        else:
            Ybits = np.sum(Y, axis=1)
        if return_XYbits:
            XYbits = (X * Y.T).toarray()
            return Xbits, Ybits, XYbits
    else:
        Xbits = np.sum(X, axis=1, keepdims=True)
        if Y is X:
            Ybits = Xbits
        else:
            Ybits = np.sum(Y, axis=1, keepdims=True)
        if return_XYbits:
            XYbits = np.dot(X, Y.T)
            return Xbits, Ybits, XYbits
    return Xbits, Ybits


def _sparse_cosine(X, Y):
    Xnorm = scipy.sqrt(X.multiply(X).sum(axis=1))
    if Y is X:
        Ynorm = Xnorm
    else:
        Ynorm = scipy.sqrt(Y.multiply(Y).sum(axis=1))
    XY = (X * Y.T).toarray()
    with np.errstate(divide="ignore"):  # handle 0 in denominator
        return np.nan_to_num(XY / (Xnorm * Ynorm.T))

@maybe_jit(nopython=True, nogil=True, cache=True)
def _dense_soergel(X, Y, S):
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
    return S

@maybe_jit(nopython=True, nogil=True, cache=True)
def _sparse_soergel(Xdata, Xindices, Xindptr, Ydata, Yindices, Yindptr, S):
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
    return S

"""Efficient comparison metrics for fingerprints and their databases.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import logging

from ..fprint import Fingerprint, BitsValueError
from ..db import FingerprintDatabase
from . import array_metrics
from . import fprint_metrics


def tanimoto(A, B=None):
    """Compute Tanimoto coefficients between fingerprints.

    Fingerprints must have same number of bits. If not bit-fingerprints,
    arrays will be cast to binary. If only one fingerprint/database is
    provided, it is compared to self.

    Parameters
    ----------
    A : Fingerprint or FingerprintDatabase
    B : Fingerprint FingerprintDatabase, optional

    Returns
    -------
    tanimoto : double or array of shape (num_fprints1, num_fprints2)
    """
    A, B = _check_item_pair(A, B, fp_type=Fingerprint)
    if isinstance(A, Fingerprint):
        return fprint_metrics.tanimoto(A, B)
    return array_metrics.tanimoto(A.array, B.array)


def soergel(A, B=None):
    """Compute Soergel similarities between fingerprints.

    Soergel similarity is the complement of the Soergel distance and is
    analogous to the Tanimoto coefficient for count/float fingerprints. For
    binary data, it is equivalent to the Tanimoto coefficient.

    Parameters
    ----------
    A : Fingerprint or FingerprintDatabase
    B : Fingerprint FingerprintDatabase, optional

    Returns
    -------
    soergel : double or array of shape (num_fprints1, num_fprints2)
    """
    A, B = _check_item_pair(A, B)
    if isinstance(A, Fingerprint):
        return fprint_metrics.soergel(A, B)
    return array_metrics.soergel(A.array, B.array)


def dice(A, B=None):
    """Compute Dice coefficients between fingerprints.

    Parameters
    ----------
    A : Fingerprint or FingerprintDatabase
    B : Fingerprint FingerprintDatabase, optional

    Returns
    -------
    dice : double or array of shape (num_fprints1, num_fprints2)
    """
    A, B = _check_item_pair(A, B, fp_type=Fingerprint)
    if isinstance(A, Fingerprint):
        return fprint_metrics.dice(A, B)
    return array_metrics.dice(A.array, B.array)


def cosine(A, B=None):
    """Compute cosine similarities between fingerprints.

    Parameters
    ----------
    A : Fingerprint or FingerprintDatabase
    B : Fingerprint FingerprintDatabase, optional

    Returns
    -------
    cosine : double or array of shape (num_fprints1, num_fprints2)
    """
    A, B = _check_item_pair(A, B)
    if isinstance(A, Fingerprint):
        return fprint_metrics.cosine(A, B)
    return array_metrics.cosine(A.array, B.array)


def pearson(A, B=None):
    """Compute Pearson correlation between fingerprints.

    Parameters
    ----------
    A : Fingerprint or FingerprintDatabase
    B : Fingerprint FingerprintDatabase, optional

    Returns
    -------
    pearson : double or array of shape (num_fprints1, num_fprints2)
    """
    A, B = _check_item_pair(A, B)
    if isinstance(A, Fingerprint):
        return fprint_metrics.pearson(A, B)
    return array_metrics.pearson(A.array, B.array)


def _check_item(item, fp_type=None, force_db=False):
    if force_db and isinstance(item, Fingerprint):
            if not fp_type:
                fp_type = item.__class__
            db = FingerprintDatabase(fp_type=fp_type)
            db.add_fingerprints([item])
            item = db
    elif fp_type and isinstance(item, FingerprintDatabase):
            logging.debug("Casting database fingerprints to {}.".format(
                fp_type.__name__))
            item = item.as_type(fp_type, copy=False)
    return item


def _check_item_pair(A, B, fp_type=None, force_db=False):
    try:
        if B is not None and A.bits != B.bits:
            raise BitsValueError(
                "Fingerprints must have same number of bits.")
    except AttributeError:
        raise TypeError("Items must be Fingerprint or FingerprintDatabase.")
    if (isinstance(A, FingerprintDatabase) or
            isinstance(B, FingerprintDatabase)):
        force_db = True
    A = _check_item(A, fp_type=fp_type, force_db=force_db)
    if B is None:
        B = A
    else:
        B = _check_item(B, fp_type=fp_type, force_db=force_db)
    return A, B

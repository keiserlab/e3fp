"""Fingerprint comparison metrics.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
from __future__ import division

import numpy as np
from ..fprint import CountFingerprint, diff_counts_dict


def tanimoto(fp1, fp2):
    """Calculate Tanimoto coefficient between fingerprints.

    Parameters
    ----------
    fp1 : Fingerprint
        Fingerprint 1
    fp2 : Fingerprint
        Fingerprint 2

    Returns
    -------
    float : Tanimoto coefficient.
    """
    try:
        intersect = np.intersect1d(
            fp1.indices, fp2.indices, assume_unique=True
        ).shape[0]
        return intersect / (fp1.bit_count + fp2.bit_count - intersect)
    except ZeroDivisionError:
        return 0.0


def soergel(fp1, fp2):
    """Calculate Soergel similarity between fingerprints.

    Soergel similarity is the complement of Soergel distance and can be
    thought of as the analog of the Tanimoto coefficient for count/float-based
    fingerprints. For `Fingerprint`, it is equivalent to the Tanimoto
    coefficient.

    Parameters
    ----------
    fp1 : Fingerprint
        Fingerprint 1
    fp2 : Fingerprint
        Fingerprint 2

    Returns
    -------
    float : Soergel similarity.

    Reference
    -------

    """
    if not (
        isinstance(fp1, CountFingerprint) and isinstance(fp2, CountFingerprint)
    ):
        return tanimoto(fp1, fp2)

    counts_diff = diff_counts_dict(fp1, fp2)
    temp = np.asarray(
        [
            (abs(counts_diff[x]), max(fp1.get_count(x), fp2.get_count(x)))
            for x in counts_diff.keys()
        ],
        dtype=np.float,
    ).T
    soergel = 1 - np.sum(temp[0, :]) / np.sum(temp[1, :])

    return soergel


def dice(fp1, fp2):
    """Calculate Dice coefficient between fingerprints.

    Parameters
    ----------
    fp1 : Fingerprint
        Fingerprint 1
    fp2 : Fingerprint
        Fingerprint 2

    Returns
    -------
    float : Dice coefficient.
    """
    try:
        intersect = np.intersect1d(
            fp1.indices, fp2.indices, assume_unique=True
        ).shape[0]
        return 2 * intersect / (fp1.bit_count + fp2.bit_count)
    except ZeroDivisionError:
        return 0.0


def cosine(fp1, fp2):
    """Calculate cosine similarity between fingerprints.

    Parameters
    ----------
    fp1 : Fingerprint
        Fingerprint 1
    fp2 : Fingerprint
        Fingerprint 2

    Returns
    -------
    float : Cosine similarity.
    """
    try:
        dot = sum(v * fp2.get_count(k) for k, v in fp1.counts.items())
        root_norm = (
            sum(v ** 2 for v in fp1.counts.values())
            * sum(v ** 2 for v in fp2.counts.values())
        ) ** 0.5
        return dot / root_norm
    except ZeroDivisionError:
        return 0.0


def pearson(fp1, fp2):
    """Calculate Pearson correlation between fingerprints.

    Parameters
    ----------
    fp1 : Fingerprint
        Fingerprint 1
    fp2 : Fingerprint
        Fingerprint 2

    Returns
    -------
    float : Pearson correlation.
    """
    try:
        dot = sum(v * fp2.get_count(k) for k, v in fp1.counts.items())
        return (dot / fp1.bits - fp1.mean() * fp2.mean()) / (
            fp1.std() * fp2.std()
        )
    except ZeroDivisionError:
        return 0.0

    # intersect = np.intersect1d(fp1.indices, fp2.indices,
    #                            assume_unique=True).shape[0]
    # return ((intersect / fp1.bits) -
    #         ((fp1.mean() * fp2.mean()) / (fp1.std() * fp2.std())))


def hamming(fp1, fp2):
    """Calculate Hamming distance between fingerprints.

    Parameters
    ----------
    fp1 : Fingerprint
        Fingerprint 1
    fp2 : Fingerprint
        Fingerprint 2

    Returns
    -------
    float : Hamming distance.
    """
    intersect = np.intersect1d(
        fp1.indices, fp2.indices, assume_unique=True
    ).shape[0]
    return fp1.bit_count + fp2.bit_count - 2 * intersect


def distance(fp1, fp2):
    """Calculate Euclidean distance between fingerprints.

    Parameters
    ----------
    fp1 : Fingerprint
        Fingerprint 1
    fp2 : Fingerprint
        Fingerprint 2

    Returns
    -------
    float : Euclidian distance.
    """
    return hamming(fp1, fp2) ** 0.5

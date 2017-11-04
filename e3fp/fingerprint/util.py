"""Utility methods and class for fingerprinting-related functions.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
from ..util import E3FPError


class E3FPInvalidFingerprintError(E3FPError, TypeError):
    """Fingerprint is incorrectly formatted."""


class E3FPMolError(E3FPError, TypeError):
    """Mol is of incorrect type."""


class E3FPBitsValueError(E3FPError, ValueError):
    """Bits value is invalid."""


class E3FPCountsError(E3FPError, ValueError):
    """Index in counts is invalid."""


class E3FPOptionError(E3FPError, ValueError):
    """Option provided is invalid."""

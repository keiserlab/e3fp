"""Classes and methods for chemical fingerprint storage and comparison.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
from __future__ import division, print_function
from collections import defaultdict

try:
    import cPickle as pkl
except ImportError:  # Python 3
    import pickle as pkl

import numpy as np
from scipy.sparse import issparse, csr_matrix

try:
    from rdkit.DataStructs.cDataStructs import ExplicitBitVect, SparseBitVect

    WITH_RDKIT = True
except ImportError:
    WITH_RDKIT = False
from python_utilities.io_tools import smart_open
from e3fp.fingerprint.util import (
    E3FPInvalidFingerprintError,
    E3FPMolError,
    E3FPBitsValueError,
    E3FPCountsError,
    E3FPOptionError,
)

# ----------------------------------------------------------------------------#
# Fingerprint Classes
# ----------------------------------------------------------------------------#

BITS_DEF = 2 ** 32
FOLD_BITS_DEF = 1024
FP_DTYPE = np.bool_
COUNT_FP_DTYPE = np.uint16
FLOAT_FP_DTYPE = np.float64
NAME_PROP_KEY = "Name"
MOL_PROP_KEY = "Mol"


def fptype_from_dtype(dtype):
    """Get corresponding fingerprint type from NumPy data type.

    Parameters
    ----------
    dtype : numpy.dtype or str
        NumPy data type.

    Returns
    -------
    class: {Fingerprint, CountFingerprint, FloatFingerprint}
        Class of fingerprint
    """
    if np.issubdtype(dtype, np.bool_):
        return Fingerprint
    elif np.issubdtype(dtype, np.integer):
        return CountFingerprint
    elif np.issubdtype(dtype, np.floating):
        return FloatFingerprint
    else:
        raise TypeError("dtype {} is invalid for fingerprint".format(dtype))


def dtype_from_fptype(fp_type):
    """Get NumPy data type from fingerprint type.

    Parameters
    ----------
    fp_type : class or Fingerprint
        Class of fingerprint

    Returns
    -------
    numpy.dtype
        NumPy data type
    """
    if isinstance(fp_type, Fingerprint):
        fp_type = fp_type.__class__
    if fp_type is Fingerprint:
        return FP_DTYPE
    elif fp_type is CountFingerprint:
        return COUNT_FP_DTYPE
    elif fp_type is FloatFingerprint:
        return FLOAT_FP_DTYPE
    else:
        raise E3FPInvalidFingerprintError(
            "fp_type {} is not a valid fp_type.".format(fp_type)
        )


def coerce_to_valid_dtype(dtype):
    """Coerce provided NumPy data type to closest fingerprint data type.

    If provided `dtype` cannot be read, default corresponding to bit
    `Fingerprint` is returned.

    Parameters
    ----------
    dtype : numpy.dtype or str
        Input NumPy data type.

    Returns
    -------
    numpy.dtype
        Output NumPy data type.
    """
    try:
        fp_type = fptype_from_dtype(dtype)
        return dtype_from_fptype(fp_type)
    except TypeError:
        return FP_DTYPE


class Fingerprint(object):
    """A fingerprint that stores indices of "on" bits.

    Parameters
    ----------
    indices : array_like of int, optional
        log2(`bits`)-bit indices in a sparse bitvector of `bits` which
        correspond to 1.
    bits : int, optional
        Number of bits in bitvector.
    level : int, optional
        Level of fingerprint, corresponding to fingerprinting iterations.
    name : str, optional
        Name of fingerprint.
    props : dict, optional
        Custom properties of fingerprint, consisting of a string keyword and
        some value.

    Attributes
    ----------
    bits : int
        Number of bits in bitvector, length of fingerprint.
    counts : dict
        Dict matching each index in `indices` to number of counts (1 for bits).
    indices : numpy.ndarray of int
        Indices of "on" bits
    level : int
        Level of fingerprint, corresponding to fingerprinting iterations.
    mol : RDKit Mol
        Mol to which fingerprint corresponds (stored in `props`).
    name : str or None
        Name of fingerprint (stored in `props`).
    props : dict
        Custom properties of fingerprint, consisting of a string keyword and
        some value.
    vector_dtype : numpy.dtype
        NumPy data type associated with fingerprint values (e.g. bits)

    See Also
    --------
    CountFingerprint: A fingerprint that stores number of occurrences of each
                      index
    FloatFingerprint: A fingerprint that stores indices of "on" bits
    e3fp.fingerprint.db.FingerprintDatabase: Efficiently store fingerprints

    Examples
    --------
    >>> import e3fp.fingerprint.fprint as fp
    >>> from e3fp.fingerprint.metrics import tanimoto
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> bits = 1024
    >>> indices = np.random.randint(0, bits, 30)
    >>> print(indices)
    [684 559 629 192 835 763 707 359   9 723 277 754 804 599  70 472 600 396
     314 705 486 551  87 174 600 849 677 537 845  72]
    >>> f = fp.Fingerprint(indices, bits=bits, level=0)
    >>> f_folded = f.fold(bits=32)
    >>> print(f_folded.indices)
    [ 0  1  3  4  5  6  7  8  9 12 13 14 15 17 18 19 21 23 24 25 26 27]
    >>> print(f_folded.to_vector(sparse=False, dtype=int))
    [1 1 0 1 1 1 1 1 1 1 0 0 1 1 1 1 0 1 1 1 0 1 0 1 1 1 1 1 0 0 0 0]
    >>> print(f_folded.to_bitstring())
    11011111110011110111010111110000
    >>> print(f_folded.to_rdkit())
    <rdkit.DataStructs.cDataStructs.ExplicitBitVect object at 0x...>
    >>> f_folded2 = fp.Fingerprint.from_indices(np.random.randint(0, bits, 30),
    ...                                         bits=bits).fold(bits=32)
    >>> print(f_folded2.indices)
    [ 0  1  3  5  7  9 10 14 15 16 17 18 19 20 23 24 25 29 30 31]
    >>> print(tanimoto(f_folded, f_folded2))
    0.5
    """

    vector_dtype = FP_DTYPE

    def __init__(
        self, indices, bits=BITS_DEF, level=-1, name=None, props={}, **kwargs
    ):
        """Initialize Fingerprint object."""
        self.reset()

        indices = np.asarray(indices, dtype=np.long)

        if np.any(indices >= bits):
            raise E3FPBitsValueError(
                "number of bits is lower than provided indices"
            )

        self.indices = np.unique(indices)
        self.bits = bits
        self.level = level
        self.update_props(props)
        if name:
            self.name = name

    def clear(self):
        """Clear temporary (and possibly large) values."""
        pass

    def reset(self):
        """Reset all values."""
        self.indices = np.asarray([], dtype=np.long)
        self.bits = 0
        self.level = -1
        self.folded_fingerprint = {}
        self.index_to_folded_index_dict = None
        self.unfolded_fingerprint = None
        self.index_to_unfolded_index_dict = None
        self.props = {}
        self.clear()

    @classmethod
    def from_indices(cls, indices, bits=BITS_DEF, level=-1, **kwargs):
        """Initialize from an array of indices.

        Parameters
        ----------
        indices : array_like of int
            Indices in a sparse bitvector of length `bits` which correspond
            to 1.
        bits : int, optional
            Number of bits in array. Indices will be log2(`bits`)-bit
            integers.
        level : int, optional
            Level of fingerprint, corresponding to fingerprinting iterations.
        name : str, optional
            Name of fingerprint.
        props : dict, optional
            Custom properties of fingerprint, consisting of a string keyword
            and some value.

        Returns
        -------
        fingerprint : Fingerprint
        """
        return cls(indices, bits=bits, level=level, **kwargs)

    @classmethod
    def from_vector(cls, vector, level=-1, **kwargs):
        """Initialize from vector.

        Parameters
        ----------
        vector : numpy.ndarray or scipy.sparse.csr_matrix
            Array of bits/counts/floats
        level : int, optional
            Level of fingerprint, corresponding to fingerprinting iterations.
        name : str, optional
            Name of fingerprint.
        props : dict, optional
            Custom properties of fingerprint, consisting of a string keyword
            and some value.

        Returns
        -------
        fingerprint : Fingerprint
        """
        if kwargs.get("bits", None) is None:
            try:
                kwargs["bits"] = vector.shape[1]
            except IndexError:
                kwargs["bits"] = vector.shape[0]
        if issparse(vector):
            indices = vector.indices.astype(np.long)
            counts = vector.data
        else:
            indices = np.asarray(np.where(vector), dtype=np.long).flatten()
            counts = vector[indices]
        counts = dict(zip(indices, counts))
        return cls.from_indices(indices, counts=counts, level=level, **kwargs)

    @classmethod
    def from_bitstring(cls, bitstring, level=-1, **kwargs):
        """Initialize from bitstring (e.g. '10010011').

        Parameters
        ----------
        bitstring : str
            String of 1s and 0s.
        level : int, optional
            Level of fingerprint, corresponding to fingerprinting iterations.
        name : str, optional
            Name of fingerprint.
        props : dict, optional
            Custom properties of fingerprint, consisting of a string keyword
            and some value.

        Returns
        -------
        fingerprint : Fingerprint
        """
        indices = [i for i, char in enumerate(bitstring) if char != "0"]
        if kwargs.get("bits", None) is None:
            kwargs["bits"] = len(bitstring)
        return cls.from_indices(indices, level=level, **kwargs)

    @classmethod
    def from_fingerprint(cls, fp, **kwargs):
        """Initialize by copying existing fingerprint.

        Parameters
        ----------
        fp : Fingerprint
            Existing fingerprint.

        Returns
        -------
        fingerprint : Fingerprint
        """
        if not isinstance(fp, Fingerprint):
            raise E3FPInvalidFingerprintError(
                "variable is %s not Fingerprint" % (fp.__class__.__name__)
            )

        new_fp = cls.from_indices(fp.indices, bits=fp.bits, level=fp.level)
        new_fp.update_props(fp.props)
        new_fp.folded_fingerprint = dict(
            [
                (k, v.__class__.from_fingerprint(v))
                for k, v in fp.folded_fingerprint.items()
            ]
        )
        return new_fp

    @classmethod
    def from_rdkit(cls, rdkit_fprint, **kwargs):
        """Initialize from RDKit fingerprint.

        If provided fingerprint is of length 2^32 - 1, assumes real
        fingerprint is of length 2^32.

        Parameters
        ----------
        rdkit_fprint : RDKit ExplicitBitVect or SparseBitVect
            Existing RDKit fingerprint.
        level : int, optional
            Level of fingerprint, corresponding to fingerprinting iterations.
        name : str, optional
            Name of fingerprint.
        props : dict, optional
            Custom properties of fingerprint, consisting of a string keyword
            and some value.

        Returns
        -------
        fingerprint : Fingerprint
        """
        if not WITH_RDKIT:
            raise ImportError("RDKit not available.")
        if not (
            isinstance(rdkit_fprint, ExplicitBitVect)
            or isinstance(rdkit_fprint, SparseBitVect)
        ):
            raise TypeError(
                "RDKit fingerprint must be a SparseBitVect or ExplicitBitVect"
            )
        bits = rdkit_fprint.GetNumBits()
        if bits == 2 ** 32 - 1:
            bits = 2 ** 32
        indices = np.asarray(rdkit_fprint.GetOnBits(), dtype=np.long)
        return cls.from_indices(indices, bits=bits, **kwargs)

    @property
    def indices(self):
        return self._indices

    @indices.setter
    def indices(self, indices):
        self._indices = np.asarray(indices, dtype=np.long)

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, level):
        self._level = level

    @property
    def bits(self):
        return self._bits

    @bits.setter
    def bits(self, bits):
        self._bits = bits

    @property
    def props(self):
        return self._props

    @props.setter
    def props(self, props):
        self._props = props

    def get_prop(self, key):
        """Get property. If not set, raise KeyError."""
        try:
            return self.props[key]
        except AttributeError:
            raise KeyError

    def set_prop(self, key, val):
        """Set property."""
        self.props[key] = val

    def update_props(self, props_dict):
        """Set multiple properties at once."""
        self.props.update(props_dict)

    @property
    def name(self):
        try:
            return self.props[NAME_PROP_KEY]
        except KeyError:
            return None

    @name.setter
    def name(self, name):
        self.props[NAME_PROP_KEY] = str(name)

    @property
    def mol(self):
        try:
            return self.props[MOL_PROP_KEY]
        except KeyError:
            return None

    @mol.setter
    def mol(self, mol):
        self.props[MOL_PROP_KEY] = mol

    @property
    def index_id_map(self):
        try:
            return self.props["index_id_map"]
        except (KeyError, AttributeError):
            return None

    @index_id_map.setter
    def index_id_map(self, index_id_map):
        self.props["index_id_map"] = index_id_map

    def to_vector(self, sparse=True, dtype=None):
        """Get vector of bits/counts/floats.

        Returns
        -------
        numpy.ndarray or scipy.sparse.csr_matrix
            Vector of bits/counts/floats
        """
        if dtype is None:
            dtype = self.vector_dtype

        counts = self.counts
        if sparse:
            try:
                return csr_matrix(
                    (
                        [counts[i] for i in self.indices],
                        ([0] * self.bit_count, self.indices),
                    ),
                    shape=(1, self.bits),
                    dtype=dtype,
                )
            except ValueError:
                raise E3FPBitsValueError(
                    "Number of bits is lower than size of indices"
                )
        else:
            bitvector = np.zeros(self.bits, dtype=dtype)
            try:
                bitvector[self.indices] = [counts[i] for i in self.indices]
                return bitvector
            except IndexError:
                raise E3FPBitsValueError(
                    "Number of bits is lower than size of indices"
                )

    def to_bitvector(self, sparse=True):
        """Get full bitvector.

        Returns
        -------
        numpy.ndarray or scipy.sparse.csr_matrix of bool : Bitvector
        """
        return self.to_vector(sparse=sparse, dtype=FP_DTYPE)

    def to_bitstring(self):
        """Get bitstring as string of 1s and 0s.

        Returns
        -------
        str : bitstring
        """
        bitvector = self.to_bitvector(sparse=False)
        return "".join(map(str, np.asarray(bitvector, dtype=np.int)))

    def to_rdkit(self):
        """Convert to RDKit fingerprint.

        If number of bits exceeds 2^31 - 1, fingerprint will be folded to
        length 2^31 - 1 before conversion.

        Returns
        -------
        rdkit_fprint : RDKit ExplicitBitVect or SparseBitVect
            Convert to bitvector used for RDKit fingerprints. If `self.bits`
            is less than 10^5, `ExplicitBitVect` is used. Otherwise,
            `SparseBitVect` is used.
        """
        if not WITH_RDKIT:
            raise ImportError("RDKit not available.")

        rdkit_fp_type = SparseBitVect
        if self.bits < 1e5:
            rdkit_fp_type = ExplicitBitVect

        # RDKit Bitvect types can't exceed 2**31 - 1 in length
        bits = min(self.bits, 2 ** 31 - 1)
        indices = self.indices % (2 ** 31 - 1)

        rdkit_fprint = rdkit_fp_type(bits)
        rdkit_fprint.SetBitsFromList(indices.tolist())
        return rdkit_fprint

    @property
    def bit_count(self):
        return self.indices.shape[0]

    @property
    def density(self):
        return self.bit_count / self.bits

    def get_count(self, index):
        """Return count index in fingerprint.

        Defaults to 1 if index in `self.indices`

        Returns
        -------
        int : Count of bit in fingerprint
        """
        if index in self.indices:
            return 1
        else:
            return 0

    @property
    def counts(self):
        return dict([(k, 1) for k in self.indices])

    def mean(self):
        """Return mean, i.e. proportion of "on" bits in fingerprint.

        Returns
        -------
        float : Mean
        """
        return self.density

    def std(self):
        """Return standard deviation of fingerprint.

        Returns
        -------
        float : Standard deviation
        """
        mean = self.mean()
        return (mean * (1 - mean)) ** 0.5

    # Folding/unfolding to a new fingerprint
    def fold(self, bits=FOLD_BITS_DEF, method=0, linked=True):
        """Return fingerprint for bitvector folded to size `bits`.

        Parameters
        ----------
        bits : int, optional
            Length of new bitvector, ideally multiple of 2.
        method : {0, 1}, optional
            Method to use for folding.

            0
                partitioning (array is divided into equal sized arrays of
                length `bits` which are bitwise combined with OR)
            1
                compression (adjacent bits pairs are combined with OR until
                length is `bits`)
        linked : bool, optional
            Link folded and unfolded fingerprints for easy referencing. Set
            to False if intending to save and want to reduce file size.

        Returns
        -------
        Fingerprint : Fingerprint of folded bitvector
        """
        if bits > self.bits:
            raise E3FPBitsValueError("folded bits greater than existing bits")
        if not np.log2(self.bits / bits).is_integer():
            raise E3FPBitsValueError(
                "existing bits divided by power of 2 does not give folded bits"
            )
        if method not in (0, 1):
            raise E3FPOptionError("method must be 0 or 1")

        if (bits, method) not in self.folded_fingerprint:
            if method == 0:
                folded_indices = self.indices % bits
            elif method == 1:
                folded_indices = self.indices / (self.bits / bits)

            self.index_to_folded_index_dict = dict(
                zip(self.indices, folded_indices)
            )
            folded_index_to_index_dict = {}
            for index, folded_index in self.index_to_folded_index_dict.items():
                folded_index_to_index_dict.setdefault(
                    folded_index, set([])
                ).add(index)

            fp = self.__class__.from_indices(
                folded_indices, bits=bits, level=self.level
            )
            fp.update_props(self.props)

            fp.index_to_unfolded_index_dict = folded_index_to_index_dict
            if self.index_id_map is not None:
                fp.index_id_map = {}
                for index, id_set in self.index_id_map.items():
                    fp.index_id_map.setdefault(
                        self.index_to_folded_index_dict[index], set()
                    ).update(id_set)

            if linked:
                fp.unfolded_fingerprint = self
                self.folded_fingerprint[(bits, method)] = fp

        assert isinstance(
            self.folded_fingerprint[(bits, method)], self.__class__
        )
        return self.folded_fingerprint[(bits, method)]

    def get_folding_index_map(self):
        """Get map of sparse indices to folded indices.

        Returns
        -------
        dict : Map of sparse index (keys) to corresponding folded index.
        """
        return self.index_to_folded_index_dict

    def unfold(self):
        """Return unfolded parent fingerprint for bitvector.

        Returns
        -------
        Fingerprint : Fingerprint of unfolded bitvector. If None, return
                      None.
        """
        return self.unfolded_fingerprint

    def get_unfolding_index_map(self):
        """Get map of sparse indices to unfolded indices.

        Returns
        -------
        dict : Map of sparse index (keys) to set of corresponding unfolded
               indices.
        """
        return self.index_to_unfolded_index_dict

    # summary magic methods
    def __repr__(self):
        return "%s(indices=%s, level=%r, bits=%r, name=%s)" % (
            self.__class__.__name__,
            repr(self.indices)
            .replace("\n", "")
            .replace(" ", "")
            .replace(",", ", "),
            self.level,
            self.bits,
            self.name,
        )

    def __str__(self):
        return self.__repr__()

    # logical/comparative magic methods
    def __eq__(self, other):
        if not isinstance(other, Fingerprint):
            raise E3FPInvalidFingerprintError(
                "variable is %s not Fingerprint" % (other.__class__.__name__)
            )

        return (
            self.level == other.level
            and self.bits == other.bits
            and self.__class__ == other.__class__
            and np.all(
                np.in1d(self.indices, other.indices, assume_unique=True)
            )
        )

    def __ne__(self, other):
        if not isinstance(other, Fingerprint):
            raise E3FPInvalidFingerprintError(
                "variable is %s not Fingerprint" % (other.__class__.__name__)
            )

        return not self.__eq__(other)

    def __add__(self, other):
        if not isinstance(other, Fingerprint):
            raise E3FPInvalidFingerprintError(
                "variable is %s not Fingerprint" % (other.__class__.__name__)
            )

        if self.bits != other.bits:
            raise E3FPBitsValueError(
                "cannot add fingerprints of different sizes"
            )

        return Fingerprint(
            np.union1d(self.indices, other.indices), bits=self.bits
        )

    def __sub__(self, other):
        if not isinstance(other, Fingerprint):
            raise E3FPInvalidFingerprintError(
                "variable is %s not Fingerprint" % (other.__class__.__name__)
            )

        if self.bits != other.bits:
            raise E3FPBitsValueError(
                "cannot subtract fingerprints of different sizes"
            )

        return Fingerprint(
            np.setdiff1d(self.indices, other.indices, assume_unique=True),
            bits=self.bits,
        )

    def __and__(self, other):
        if not isinstance(other, Fingerprint):
            raise E3FPInvalidFingerprintError(
                "variable is %s not Fingerprint" % (other.__class__.__name__)
            )

        if self.bits != other.bits:
            raise E3FPBitsValueError(
                "cannot compare fingerprints of different sizes"
            )

        return Fingerprint(
            np.intersect1d(self.indices, other.indices, assume_unique=True),
            bits=self.bits,
        )

    def __or__(self, other):
        if not isinstance(other, Fingerprint):
            raise E3FPInvalidFingerprintError(
                "variable is %s not Fingerprint" % (other.__class__.__name__)
            )

        if self.bits != other.bits:
            raise E3FPBitsValueError(
                "cannot compare fingerprints of different sizes"
            )

        return Fingerprint(
            np.union1d(self.indices, other.indices), bits=self.bits
        )

    def __xor__(self, other):
        if not isinstance(other, Fingerprint):
            raise E3FPInvalidFingerprintError(
                "variable is %s not Fingerprint" % (other.__class__.__name__)
            )

        if self.bits != other.bits:
            raise E3FPBitsValueError(
                "cannot compare fingerprints of different sizes"
            )

        return Fingerprint(
            np.setxor1d(self.indices, other.indices, assume_unique=True),
            bits=self.bits,
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __rand__(self, other):
        return self.__and__(other)

    def __ror__(self, other):
        return self.__ror__(other)

    def __rxor__(self, other):
        return self.__rxor__(other)

    def __iadd__(self, other):
        return self.__add__(other)

    def __isub__(self, other):
        return self.__sub__(other)

    def __iand__(self, other):
        return self.__and__(other)

    def __ior__(self, other):
        return self.__ror__(other)

    def __ixor__(self, other):
        return self.__rxor__(other)

    # iterable magic methods
    def __len__(self):
        return self.bits

    def __getitem__(self, key):
        if type(key) is not int:
            raise TypeError
        elif self.indices is None:
            raise KeyError
        elif key > self.bits:
            raise KeyError
        elif key < -self.bits:
            raise KeyError
        else:
            return key in self.indices

    # pickle magic methods, reduces size of fingerprint file
    def __getstate__(self):
        return dict([(k, v) for k, v in self.__dict__.items()])

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.clear()


class CountFingerprint(Fingerprint):
    """A fingerprint that stores number of occurrences of each index.

    Parameters
    ----------
    indices : array_like of int, optional
        log2(`bits`)-bit indices in a sparse vector, corresponding to positions
        with counts greater than 0. If not provided, `counts` must be provided.
    counts : dict, optional
        Dict matching each index in `indices` to number of counts. All counts
        default to 1 if not provided.
    bits : int, optional
        Number of bits in bitvector.
    level : int, optional
        Level of fingerprint, corresponding to fingerprinting iterations.
    name : str, optional
        Name of fingerprint.
    props : dict, optional
        Custom properties of fingerprint, consisting of a string keyword and
        some value.

    Attributes
    ----------
    bits : int
        Number of bits in bitvector, length of fingerprint.
    counts : dict
        Dict matching each index in `indices` to number of counts.
    indices : numpy.ndarray of int
        Indices of fingerprint with counts greater than 0.
    level : int
        Level of fingerprint, corresponding to fingerprinting iterations.
    mol : RDKit Mol
        Mol to which fingerprint corresponds (stored in `props`).
    name : str or None
        Name of fingerprint (stored in `props`).
    props : dict
        Custom properties of fingerprint, consisting of a string keyword and
        some value.
    vector_dtype : numpy.dtype
        NumPy data type associated with fingerprint values (e.g. bits)

    See Also
    --------
    Fingerprint: A fingerprint that stores indices of "on" bits
    FloatFingerprint: A fingerprint that stores float counts

    Examples
    --------
    >>> import e3fp.fingerprint.fprint as fp
    >>> from e3fp.fingerprint.metrics import soergel
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> bits = 1024
    >>> indices = np.random.randint(0, bits, 30)
    >>> print(indices)
    [ 37 235 908  72 767 905 715 645 847 960 144 129 972 583 749 508 390 281
     178 276 254 357 914 468 907 252 490 668 925 398]
    >>> counts = dict(zip(indices,
    ...                   np.random.randint(1, 100, indices.shape[0])))
    >>> print(sorted(counts.items()))
    [(37, 51), (72, 88), (129, 62), ..., (925, 50), (960, 8), (972, 23)]
    >>> f = fp.CountFingerprint(indices, counts=counts, bits=bits, level=0)
    >>> f_folded = f.fold(bits=32)
    >>> print(sorted(f_folded.counts.items()))
    [(0, 8), (1, 62), (5, 113), ..., (29, 50), (30, 14), (31, 95)]
    >>> print(f_folded.to_vector(sparse=False, dtype=int))
    [  8  62   0   0   0 113  61  58  88  97  71 228 111   2  58  10  64   0
      82   0 120   0   0   0   0  82   0   0  27  50  14  95]
    >>> fp.Fingerprint.from_fingerprint(f_folded)
    Fingerprint(indices=array([0, 1, ...]), level=0, bits=32, name=None)
    >>> indices2 = np.random.randint(0, bits, 30)
    >>> counts2 = dict(zip(indices2,
    ...                    np.random.randint(1, 100, indices.shape[0])))
    >>> f_folded2 = fp.CountFingerprint.from_indices(indices2, counts=counts2,
    ...                                              bits=bits).fold(bits=32)
    >>> print(sorted(f_folded2.counts.items()))
    [(0, 93), (2, 33), (3, 106), ..., (25, 129), (26, 89), (30, 53)]
    >>> print(soergel(f_folded, f_folded2))
    0.17492946392...
    """

    vector_dtype = COUNT_FP_DTYPE

    def __init__(
        self,
        indices=None,
        counts=None,
        bits=BITS_DEF,
        level=-1,
        name=None,
        props={},
        **kwargs
    ):
        """Initialize."""
        if indices is None and counts is None:
            raise E3FPOptionError("indices or counts must be specified")

        self.reset()

        if indices is not None:
            indices = np.asarray(indices, dtype=np.long)

            if np.any(indices >= bits):
                raise E3FPBitsValueError(
                    "number of bits is lower than provided indices"
                )

            if counts is None:
                indices, counts = np.unique(indices, return_counts=True)
                counts = dict(zip(indices, counts))
            else:
                indices = np.unique(indices)
                if not np.all([x in indices for x in counts]):
                    raise E3FPCountsError(
                        "At least one index in `counts` is not in `indices`."
                    )
                if len(set(indices).symmetric_difference(counts)) > 0:
                    raise E3FPCountsError(
                        "At least one index in `indices` is not in `counts`."
                    )

        else:
            indices = np.asarray(sorted(counts.keys()), dtype=np.long)

            if np.any(indices >= bits):
                raise E3FPBitsValueError(
                    "number of bits is lower than provided indices"
                )

        self.indices = indices
        self.counts = counts
        self.bits = bits
        self.level = level
        if name:
            self.props[NAME_PROP_KEY] = name
        self.update_props(props)

    @classmethod
    def from_indices(
        cls, indices, counts=None, bits=BITS_DEF, level=-1, **kwargs
    ):
        """Initialize from an array of indices.

        Parameters
        ----------
        indices : array_like of int, optional
            Indices in a sparse bitvector of length `bits` which correspond to
            1.
        counts : dict, optional
            Dictionary mapping sparse indices to counts.
        bits : int, optional
            Number of bits in array. Indices will be log2(`bits`)-bit
            integers.
        level : int, optional
            Level of fingerprint, corresponding to fingerprinting iterations.
        name : str, optional
            Name of fingerprint.
        props : dict, optional
            Custom properties of fingerprint, consisting of a string keyword
            and some value.

        Returns
        -------
        fingerprint : CountFingerprint
        """
        return cls(indices, counts=counts, bits=bits, level=level, **kwargs)

    @classmethod
    def from_counts(cls, counts, bits=BITS_DEF, level=-1, **kwargs):
        """Initialize from an array of indices.

        Parameters
        ----------
        counts : dict
            Dictionary mapping sparse indices to counts.
        bits : int, optional
            Number of bits in array. Indices will be log2(`bits`)-bit
            integers.
        level : int, optional
            Level of fingerprint, corresponding to fingerprinting iterations.
        name : str, optional
            Name of fingerprint.
        props : dict, optional
            Custom properties of fingerprint, consisting of a string keyword
            and some value.

        Returns
        -------
        fingerprint : CountFingerprint
        """
        return cls(counts=counts, bits=bits, level=level, **kwargs)

    @classmethod
    def from_fingerprint(cls, fp, **kwargs):
        """Initialize by copying existing fingerprint.

        Parameters
        ----------
        fp : Fingerprint
            Existing fingerprint.
        name : str, optional
            Name of fingerprint.
        props : dict, optional
            Custom properties of fingerprint, consisting of a string keyword
            and some value.

        Returns
        -------
        fingerprint : Fingerprint
        """
        if not isinstance(fp, Fingerprint):
            raise E3FPInvalidFingerprintError(
                "variable is %s not Fingerprint" % (fp.__class__.__name__)
            )

        counts = dict([(i, c) for i, c in fp.counts.items() if c > 0])
        new_fp = cls.from_counts(counts, bits=fp.bits, level=fp.level)
        new_fp.update_props(fp.props)
        new_fp.folded_fingerprint = dict(
            [
                (k, v.__class__.from_fingerprint(v))
                for k, v in fp.folded_fingerprint.items()
            ]
        )
        return new_fp

    def reset(self, *args, **kwargs):
        """Reset all values."""
        super(CountFingerprint, self).reset(*args, **kwargs)
        self.counts = {}

    def get_count(self, index):
        """Return count index in fingerprint.

        Returns
        -------
        int : Count of index in fingerprint
        """
        return self.counts.get(index, 0)

    @property
    def counts(self):
        return self._counts

    @counts.setter
    def counts(self, counts):
        self._counts = dict([(k, int(v)) for k, v in counts.items()])

    def mean(self):
        """Return mean of counts.

        Returns
        -------
        float : Mean
        """
        return sum(self._counts.values()) / self.bits

    def std(self):
        """Return standard deviation of fingerprint.

        Returns
        -------
        float : Standard deviation
        """
        mean = self.mean()
        return (
            sum(v ** 2 for v in self._counts.values()) / self.bits - mean ** 2
        ) ** 0.5

    def fold(self, *args, **kwargs):
        """Fold fingerprint while considering counts.

        Optionally, provide a function to reduce colliding counts.

        Parameters
        ----------
        bits : int, optional
            Length of new bitvector, ideally multiple of 2.
        method : {0, 1}, optional
            Method to use for folding.

            0
                partitioning (array is divided into equal sized arrays of
                length `bits` which are bitwise combined with `counts_method`)
            1
                compression (adjacent bits pairs are combined with
                `counts_method` until length is `bits`)
        linked : bool, optional
            Link folded and unfolded fingerprints for easy referencing. Set to
            False if intending to save and want to reduce file size.
        counts_method : function, optional
            Function for combining counts. Default is summation.

        Returns
        -------
        CountFingerprint : Fingerprint of folded vector
        """
        counts_method = kwargs.get("counts_method", sum)

        fp = super(CountFingerprint, self).fold(*args, **kwargs)
        counts = dict(
            [
                (fold_ind, counts_method([self.get_count(x) for x in ind_set]))
                for fold_ind, ind_set in fp.index_to_unfolded_index_dict.items()
            ]
        )
        fp.counts = counts
        return fp

    # summary magic methods
    def __repr__(self):
        return "%s(counts=%r, level=%r, bits=%r, name=%s)" % (
            self.__class__.__name__,
            self.counts,
            self.level,
            self.bits,
            self.name,
        )

    # logical/comparative magic methods
    def __eq__(self, other):
        if not isinstance(other, CountFingerprint):
            raise E3FPInvalidFingerprintError(
                "variable is %s not CountFingerprint"
                % (other.__class__.__name__)
            )

        return (
            self.level == other.level
            and self.bits == other.bits
            and self.counts == other.counts
            and self.__class__ == other.__class__
        )

    def __ne__(self, other):
        if not isinstance(other, Fingerprint):
            raise E3FPInvalidFingerprintError(
                "variable is %s not CountFingerprint"
                % (other.__class__.__name__)
            )

        return not self.__eq__(other)

    def __add__(self, other):
        if not isinstance(other, CountFingerprint):
            raise E3FPInvalidFingerprintError(
                "variable is not CountFingerprint."
            )

        if self.bits != other.bits:
            raise E3FPBitsValueError(
                "cannot add fingerprints of different sizes"
            )

        if self.level == other.level:
            level = self.level
        else:
            level = -1

        new_counts = self.counts.copy()
        for k, v in list(other.counts.items()):
            new_counts[k] = new_counts.get(k, 0) + v

        new_indices = np.asarray(list(new_counts.keys()), dtype=np.long)

        if other.__class__ is FloatFingerprint:
            new_class = FloatFingerprint
        else:
            new_class = self.__class__

        return new_class(
            new_indices, counts=new_counts, bits=self.bits, level=level
        )

    def __sub__(self, other):
        if not isinstance(other, CountFingerprint):
            raise E3FPInvalidFingerprintError(
                "variable is not CountFingerprint."
            )

        if self.bits != other.bits:
            raise E3FPBitsValueError(
                "cannot subtract fingerprints of different sizes"
            )

        if self.level == other.level:
            level = self.level
        else:
            level = -1

        new_counts = self.counts.copy()
        for k, v in other.counts.items():
            new_counts[k] = new_counts.get(k, 0) - v

        new_indices = np.asarray(new_counts.keys(), dtype=np.long)

        if other.__class__ is FloatFingerprint:
            new_class = FloatFingerprint
        else:
            new_class = self.__class__

        return new_class(
            new_indices, counts=new_counts, bits=self.bits, level=level
        )

    def __floordiv__(self, x):
        cf = CountFingerprint.from_fingerprint(self)
        cf.counts = dict(
            [(k, int(v / x)) for k, v in self.counts.items() if v >= x]
        )
        return cf

    def __div__(self, x):
        x = float(x)
        cf = FloatFingerprint.from_fingerprint(self)
        cf.counts = dict([(k, v / x) for k, v in self.counts.items()])
        return cf

    def __truediv__(self, x):
        return self.__div__(x)

    def __mul__(self, x):
        cf = self.__class__.from_fingerprint(self)
        cf.counts = dict([(k, v * float(x)) for k, v in self.counts.items()])
        return cf

    def __rfloordiv__(self, x):
        return self.__floordiv__(x)

    def __rdiv__(self, x):
        return self.__div__(x)

    def __rtruediv__(self, x):
        return self.__truediv__(x)

    def __rmul__(self, x):
        return self.__mul__(x)

    def __ifloordiv__(self, x):
        return self.__floordiv__(x)

    def __idiv__(self, x):
        return self.__div__(x)

    def __itruediv__(self, x):
        return self.__truediv__(x)

    def __imul__(self, x):
        return self.__mul__(x)

    # iterable magic mathods
    def __len__(self):
        return self.bits

    def __getitem__(self, key):
        if type(key) is not int:
            raise TypeError
        elif self.indices is None:
            raise KeyError
        elif key > self.bits:
            raise KeyError
        elif key < -self.bits:
            raise KeyError
        else:
            return key in self.indices

    # pickle magic methods, reduces size of fingerprint
    def __getstate__(self):
        return dict(
            [(k, v) for k, v in self.__dict__.items() if k not in ("indices",)]
        )

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.indices = sorted(self.counts.keys())
        self.clear()


class FloatFingerprint(CountFingerprint):
    """A Fingerprint that stores float counts.

    Nearly identical to `CountFingerprint`. Mainly a naming convention, but
    count values are stored as floats.


    See Also
    --------
    Fingerprint: A fingerprint that stores indices of "on" bits
    CountFingerprint: A fingerprint that stores number of occurrences of each
                      index
    """

    vector_dtype = FLOAT_FP_DTYPE

    @property
    def counts(self):
        return self._counts

    @counts.setter
    def counts(self, counts):
        self._counts = dict([(k, float(v)) for k, v in counts.items()])


# ----------------------------------------------------------------------------#
# Serialization Methods
# ----------------------------------------------------------------------------#


def load(f, update_structure=True):
    """Load `Fingerprint` object from file.

    Parameters
    ----------
    f : str or File
        File name or file-like object to load file from.
    update_structure : bool, optional
        Attempt to update the class structure by initializing a new, shiny
        fingerprint from each fingerprint in the file. Useful for guaranteeing
        that old, dusty fingerprints are always upgradeable.

    Returns
    -------
    Fingerprint : Pickled fingerprint.

    See Also
    --------
    loadz, save
    """
    fps = _load(f, update_structure)
    if len(fps) == 0:
        return None
    else:
        return fps[0]


def loadz(f, update_structure=True):
    """Load `Fingerprint` objects from file.

    Parameters
    ----------
    f : str or File
        File name or file-like object to load file from.
    update_structure : bool, optional
        Attempt to update the class structure by initializing a new, shiny
        fingerprint from each fingerprint in the file. Useful for guaranteeing
        that old, dusty fingerprints are always upgradeable. If this doesn't
        work, falls back to the original saved fingerprint.

    Returns
    -------
    list of Fingerprint : Fingerprints in pickle.

    See Also
    --------
    load, savez
    """
    return _load(f, update_structure)


def _load(f, update_structure=True):
    fps = []
    with smart_open(f, "r") as fh:
        try:
            while True:
                fp = pkl.load(fh)
                if update_structure:
                    try:
                        fps.append(fp.__class__.from_fingerprint(fp))
                    except AttributeError:
                        fps.append(fp)
                else:
                    fps.append(fp)
        except EOFError:
            pass

    return fps


def save(f, fp, **kwargs):
    """Save `Fingerprint` object to file.

    Parameters
    ----------
    f : str or File
        filename `str` or file-like object to save file to
    fp : Fingerprint
        Fingerprint to save to file
    protocol : {0, 1, 2, None}, optional
        Pickle protocol to use. If None, highest available protocol is used.
        This will not affect fingerprint loading.

    Returns
    -------
    bool : Success or fail

    See Also
    --------
    savez, load
    """
    return _save(f, fp, **kwargs)


def savez(f, *fps, **kwargs):
    """Save multiple `Fingerprint` objects to file.

    Parameters
    ----------
    f : str or File
        filename `str` or file-like object to save file to
    fps : list of Fingerprint
        List of Fingerprints to save to file
    protocol : {0, 1, 2, None}, optional
        Pickle protocol to use. If None, highest available protocol is used.
        This will not affect fingerprint loading.

    Returns
    -------
    bool : Success or fail

    See Also
    --------
    save, loadz
    """
    return _save(f, *fps, **kwargs)


def _save(f, *fps, **kwargs):
    default_dict = {"protocol": None}
    default_dict.update(kwargs)
    protocol = default_dict["protocol"]

    with smart_open(f, "w") as fh:
        if protocol is None:
            protocol = pkl.HIGHEST_PROTOCOL

        for fp in fps:
            pkl.dump(fp, fh, protocol)

    return True


def add(fprints, weights=None):
    """Add fingerprints by count to new `CountFingerprint`.

    If any of the fingerprints are `FloatFingerprint`, resulting fingerprint is
    likewise a `FloatFingerprint`. Otherwise, resulting fingerprint is
    `CountFingerprint`.

    Parameters
    ----------
    fprints : iterable of Fingerprint
        Fingerprints to be added by count.
    weights : iterable of float
        Weights for weighted sum. Results in `FloatFingerprint` output.

    Returns
    -------
    CountFingerprint or FloatFingerprint
        Fingerprint with counts as sum of counts in `fprints`.

    See Also
    --------
    mean
    """
    if len(fprints) == 0:
        return None

    if weights is None:
        new_counts = sum_counts_dict(*fprints)
        for fprint in fprints:
            if isinstance(fprint, FloatFingerprint):
                new_class = FloatFingerprint
                break
        else:
            new_class = CountFingerprint
    elif len(weights) != len(fprints):
        raise ValueError(
            "Number of fingerprints and weights must be the same."
        )
    else:
        new_counts = sum_counts_dict(*fprints, weights=weights)
        new_class = FloatFingerprint

    new_indices = np.asarray(sorted(new_counts.keys()), dtype=np.long)

    return new_class(
        new_indices,
        counts=new_counts,
        bits=fprints[0].bits,
        level=fprints[0].level,
    )


def mean(fprints, weights=None):
    """Average fingerprints to generate `FloatFingerprint`.

    Parameters
    ----------
    fprints : iterable of Fingerprint
        Fingerprints to be added by count.
    weights : array_like of float, optional
        Weights for weighted mean. Weights are normalized to a sum of 1.

    Returns
    -------
    FloatFingerprint : Fingerprint with float counts as average of counts in
                       `fprints`.
    """
    if weights is not None:
        weights = np.asarray(weights)
        weight_sum = np.sum(weights)
        if weight_sum == 0.0:
            raise ValueError("Sum of weights is 0.")
        weights = weights / weight_sum
        return add(fprints, weights=weights)
    else:
        return add(fprints) / len(fprints)


def sum_counts_dict(*fprints, **kwargs):
    """Given fingerprints, return sum of their counts dicts.

    If an optional `weights` iterable of the same length as `fprints` is
    provided, the weighted sum is returned.

    Parameters
    ----------
    *fprints
        One or more `Fingerprint` objects
    weights : iterable of float, optional
        Weights for weighted mean. Weights are normalized to a sum of 1.

    Returns
    -------
    dict : Dict of non-zero count indices in any of the `fprints` with value
           as sum of counts.

    See Also
    --------
    diff_counts_dict
    """
    counts_sum = defaultdict(int)
    if "weights" not in kwargs:
        for fprint in fprints:
            for k, v in fprint.counts.items():
                counts_sum[k] += v
    else:
        weights = kwargs["weights"]
        for (fprint, weight) in zip(fprints, weights):
            for k, v in fprint.counts.items():
                counts_sum[k] += v * weight
    return counts_sum


def diff_counts_dict(fp1, fp2, only_positive=False):
    """Given two fingerprints, returns difference of their counts dicts.

    Parameters
    ----------
    fp1, fp2 : Fingerprint
        `Fingerprint` objects, `fp2` subtracted from `fp1`.
    only_positive : bool, optional
        Return only positive counts, negative being thresholded to 0.

    Returns
    -------
    counts_diff : dict
        Count indices in either `fp1` or `fp2` with value as diff of counts.

    See Also
    --------
    sum_counts_dict
    """
    counts_diff = fp1.counts.copy()
    for k, v in fp2.counts.items():
        counts_diff[k] = counts_diff.get(k, 0) - v
        if only_positive and counts_diff[k] < 0:
            del counts_diff[k]
    return counts_diff

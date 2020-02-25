"""Database for accessing and serializing fingerprints.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
from __future__ import division
from collections import defaultdict

try:
    import cPickle as pkl
except ImportError:  # Python 3
    import pickle as pkl
import logging
import warnings

import numpy as np
import scipy
from scipy.sparse import vstack, csr_matrix
from python_utilities.io_tools import smart_open
from ..util import deprecated, E3FPEfficiencyWarning
from .fprint import (
    Fingerprint,
    CountFingerprint,
    FloatFingerprint,
    fptype_from_dtype,
    dtype_from_fptype,
    NAME_PROP_KEY,
)
from .util import E3FPBitsValueError, E3FPInvalidFingerprintError


class FingerprintDatabase(object):
    """Efficiently build, access, compare, and save fingerprints.

    Fingerprints must have the same values of `bits` and `level`.
    Additionally, all fingerprints will be cast to the type of fingerprint
    passed to the database upon instantiation.

    Parameters
    ----------
    fp_type : type, optional
        Type of fingerprint (`Fingerprint`, `CountFingerprint`,
        `FloatFingerprint`).
    level : int, optional
        Level, or number of iterations used during fingerprinting.
    name : str, optional
        Name of database.

    Attributes
    ----------
    array : scipy.sparse.csr_matrix
        Sparse matrix with dimensions N x M, where M is `bits`, and M is
        `fp_num`.
    bits : int
        Number of bits (length) of fingerprints.
    fp_names : list of str
        Names of fingerprints.
    fp_names_to_indices : dict
        Map from fingerprint name to row indices of `array`.
    fp_num : int
        Number of fingerprints in database.
    fp_type : type
        Type of fingerprint (`Fingerprint`, `CountFingerprint`,
        `FloatFingerprint`)
    level : int
        Level, or number of iterations used during fingerprinting.
    name : str
        Name of database
    props : dict
        Dict with keys specifying names of fingerprint properties and values
        corresponding to array of values.

    Notes
    -----
    Since most fingerprints are very sparse length-wise, `FingerprintDatabase`
    is implemented as a wrapper around a `scipy.sparse.csr_matrix` for
    efficient memory usage. This provides easy access to underlying data for
    tight integration with NumPy/SciPy and machine learning packages while
    simultaneously providing several fingerprint-specific features.

    See Also
    --------
    e3fp.fingerprint.fprint.Fingerprint: A fingerprint that stores indices of
                                         "on" bits

    Examples
    --------
    >>> from e3fp.fingerprint.db import FingerprintDatabase
    >>> from e3fp.fingerprint.fprint import Fingerprint
    >>> import numpy as np
    >>> np.random.seed(2)
    >>> db = FingerprintDatabase(fp_type=Fingerprint, name="TestDB")
    >>> print(db)
    FingerprintDatabase[name: TestDB, fp_type: Fingerprint, level: -1, bits: None, fp_num: 0]
    >>> bvs = (np.random.uniform(size=(3, 1024)) > .9).astype(bool)
    >>> fps = [Fingerprint.from_vector(bvs[i, :], name="fp" + str(i))
    ...        for i in range(bvs.shape[0])]
    >>> db.add_fingerprints(fps)
    >>> print(db)
    FingerprintDatabase[name: TestDB, fp_type: Fingerprint, level: -1, bits: 1024, fp_num: 3]

    The contained fingerprints may be accessed by index or name.

    >>> db[0]
    Fingerprint(indices=array([40, ..., 1012]), level=-1, bits=1024, name=fp0)
    >>> db['fp2']
    [Fingerprint(indices=array([0, ..., 1013]), level=-1, bits=1024, name=fp2)]

    Alternatively, the underlying `scipy.sparse.csr_matrix` may be accessed.

    >>> db.array
    <3x1024 sparse matrix of type '<... 'numpy.bool_'>'
    ...with 327 stored elements in Compressed Sparse Row format>
    >>> db.array.toarray()
    array([[False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           [ True, False, False, ..., False, False, False]])

    Fingerprint properties may be stored in the database.

    >>> db.set_prop("prop", np.arange(3))

    The database can be efficiently stored and loaded.

    >>> db.savez("/tmp/test_db.fpz")
    >>> db = FingerprintDatabase.load("/tmp/test_db.fpz")
    >>> print(db)
    FingerprintDatabase[name: TestDB, fp_type: Fingerprint, level: -1, bits: 1024, fp_num: 3]

    Various comparison metrics in `e3fp.fingerprint.metrics` can operate
    efficiently directly on databases

    >>> from e3fp.fingerprint.metrics import tanimoto, dice, cosine
    >>> tanimoto(db, db)
    array([[1.        , 0.0591133 , 0.04245283],
           [0.0591133 , 1.        , 0.0531401 ],
           [0.04245283, 0.0531401 , 1.        ]])
    >>> dice(db, db)
    array([[1.        , 0.11162791, 0.08144796],
           [0.11162791, 1.        , 0.10091743],
           [0.08144796, 0.10091743, 1.        ]])
    >>> cosine(db, db)
    array([[1.        , 0.11163878, 0.08145547],
           [0.11163878, 1.        , 0.10095568],
           [0.08145547, 0.10095568, 1.        ]])
    """

    def __init__(self, fp_type=Fingerprint, level=-1, name=None):
        if fp_type not in (Fingerprint, CountFingerprint, FloatFingerprint):
            raise TypeError(
                "{} is not a valid fingerprint type".format(fp_type)
            )
        self.name = name
        self.fp_type = fp_type
        self.level = level
        self.array = None
        self.fp_names = []
        self.fp_names_to_indices = defaultdict(list)
        self.props = {}

    def add_fingerprints(self, fprints):
        """Add fingerprints to database.

        Parameters
        ----------
        fprints : iterable of Fingerprint
            Fingerprints to add to database
        """
        self._check_fingerprints_are_valid(fprints)

        dtype = self.fp_type.vector_dtype

        if self.fp_num > 0:
            prop_names = self.props.keys()
        else:
            prop_names = [
                k for k in fprints[0].props.keys() if k != NAME_PROP_KEY
            ]

        new_rows = []
        new_names = []
        new_props = {x: [] for x in prop_names}
        for fprint in fprints:
            new_rows.append(fprint.to_vector(sparse=True, dtype=dtype))
            new_names.append(fprint.name)
            for prop_name in prop_names:
                new_props[prop_name].append(fprint.get_prop(prop_name))

        try:
            old_fp_num = self.fp_num
            self.array = vstack([self.array] + list(new_rows))
        except (AttributeError, ValueError):  # array not yet defined
            old_fp_num = 0
            self.array = vstack(new_rows)
        self.array = self.array.tocsr()
        del new_rows

        self.fp_names += new_names
        self.update_names_map(new_names=new_names, offset=old_fp_num)
        self.update_props(new_props, append=True)

    def update_names_map(self, new_names=None, offset=0):
        """Update map of fingerprint names to row indices of `self.array`.

        Parameters
        ----------
        new_names : iterable of str, optional
            Names to add to map. If None, map is completely rebuilt.
        offset : int, optional
            Number of rows before new rows.
        """
        if new_names is None:
            new_names = self.fp_names
        for i, name in enumerate(new_names):
            self.fp_names_to_indices[name].append(i + offset)

    def update_props(self, props_dict, append=False, check_length=True):
        """Set multiple properties at once.

        Parameters
        ----------
        props_dict : dict
            Dict of properties. Values must be array-like of length `fp_num`.
        append : bool, optional
            Append values to those already in database. By default,
            properties are overwritten if already present.
        check_length : bool, optional
            Check to ensure number of properties match number of fingerprints
            already in database. This should only be set to False for
            temporary iterative updating.
        """
        for prop_name, prop_vals in props_dict.items():
            if append and prop_name in self.props:
                prop_vals = np.append(self.get_prop(prop_name), prop_vals)
            self.set_prop(prop_name, prop_vals, check_length=check_length)

    def get_subset(self, fp_names, name=None):
        """Get database with subset of fingerprints.

        Parameters
        ----------
        fp_names : list of str
            List of fingerprint names to include in new db.
        name : str, optional
            Name of database
        """
        try:
            indices, fp_names = zip(
                *[
                    (y, x)
                    for x in fp_names
                    for y in self.fp_names_to_indices[x]
                ]
            )
        except KeyError:
            raise ValueError(
                "Not all provided fingerprint names are in database."
            )
        array = self.array[indices, :]
        props = {k: v[list(indices)] for k, v in self.props.items()}
        return FingerprintDatabase.from_array(
            array,
            fp_names=fp_names,
            fp_type=self.fp_type,
            level=self.level,
            name=name,
            props=props,
        )

    def get_density(self, index=None):
        """Get percentage of fingerprints with 'on' bit at position.

        Parameters
        ----------
        index : int or None, optional
            Index to bit for which to return positional density. If None,
            density for whole database is returned.

        Returns
        -------
        float
            Density of 'on' position in database
        """
        if index is not None:
            if not isinstance(index, int):
                raise TypeError("Index must be an integer")
            return (self.array.indices == index).sum() / self.fp_num
        return self.array.nnz / (self.bits * self.fp_num)

    def as_type(self, fp_type, copy=False):
        """Get database with fingerprint type `fp_type`.

        Parameters
        ----------
        fp_type : type
            Type of fingerprint (`Fingerprint`, `CountFingerprint`,
            `FloatFingerprint`)
        copy : bool, optional
            Force copy of database. If False, if database is already of
            requested type, no copy is made.

        Returns
        -------
        FingerprintDatabase
            Database coerced to fingerprint type of `fp_type`.
        """
        if fp_type is self.fp_type and not copy:
            return self
        return FingerprintDatabase.from_array(
            self.array,
            fp_names=self.fp_names,
            fp_type=fp_type,
            level=self.level,
            name=self.name,
            props=self.props,
        )

    def fold(self, bits, fp_type=None, name=None):
        """Get copy of database folded to specified bit length.

        Parameters
        ----------
        bits : int
            Number of bits to which to fold database.
        fp_type : type or None, optional
            Type of fingerprint (Fingerprint, CountFingerprint,
            FloatFingerprint). Defaults to same type.
        name : str, optional
            Name of database

        Returns
        -------
        FingerprintDatabase
            Database folded to specified length.

        Raises
        ------
        BitsValueError
            If `bits` is greater than the length of the database or database
            cannot be evenly folded to length `bits`.
        """
        if bits > self.bits:
            raise E3FPBitsValueError("Folded bits greater than existing bits")
        if not np.log2(self.bits / bits).is_integer():
            raise E3FPBitsValueError(
                "Existing bits divided by power of 2 does not give folded bits"
            )
        if fp_type is None:
            fp_type = self.fp_type
        dtype = dtype_from_fptype(fp_type)
        if name is None:
            name = self.name
        fold_arr = csr_matrix(
            (self.array.data, self.array.indices % bits, self.array.indptr),
            shape=self.array.shape,
        )
        fold_arr.sum_duplicates()
        fold_arr = fold_arr[:, :bits].tocsr()
        fold_arr.data = fold_arr.data.astype(dtype, copy=False)
        return self.from_array(
            fold_arr,
            fp_names=self.fp_names,
            fp_type=fp_type,
            level=self.level,
            name=name,
            props=self.props,
        )

    @classmethod
    def from_array(
        cls, array, fp_names, fp_type=None, level=-1, name=None, props={}
    ):
        """Instantiate from array.

        Parameters
        ----------
        array : numpy.ndarray or scipy.sparse.csr_matrix
            Sparse matrix with dimensions `N` x `M`, where `M` is the number
            of bits in the fingerprints.
        fp_names : list of str
            `N` names of fingerprints in `array`.
        fp_type : type, optional
            Type of fingerprint (Fingerprint, CountFingerprint,
            FloatFingerprint).
        level : int, optional
            Level, or number of iterations used during fingerprinting.
        name : str or None, optional
            Name of database.
        props : dict, optional
            Dict with keys specifying names of fingerprint properties and
            values corresponding to length `N` array of values.

        Returns
        -------
        FingerprintDatabase
            Database containing fingerprints in `array`.
        """
        dtype = array.dtype
        if fp_type is None:
            try:
                fp_type = fptype_from_dtype(dtype)
            except TypeError:
                logging.warning(
                    (
                        "`fp_type` not provided and array dtype {} does not "
                        "match fingerprint-associated dtype. Defaulting to "
                        "binary `Fingerprint.`"
                    ).format(dtype)
                )
                fp_type = Fingerprint
                dtype = dtype_from_fptype(fp_type)
        else:
            dtype = dtype_from_fptype(fp_type)
        db = cls(fp_type=fp_type, level=level, name=name)
        db.array = csr_matrix(array, dtype=dtype)
        db.fp_names = list(fp_names)
        db.update_names_map()
        db.update_props(props)
        return db

    @deprecated("1.2", msg="Use `savez` instead.")
    def save(self, fn="fingerprints.fps.bz2"):
        """Save database to file.

        Parameters
        ----------
        fn : str, optional
            Filename or basename if extension does not include '.fps'
        """
        if ".fps" not in fn:
            fn += ".fps.bz2"
        with smart_open(fn, "w") as f:
            pkl.dump(self, f)

    def savez(self, fn="fingerprints.fpz"):
        """Save database to file.

        Database is serialized using `numpy.savez_compressed`.

        Parameters
        ----------
        fn : str, optional
            Filename or basename if extension is not '.fpz'
        """
        if not fn.endswith(".fpz"):
            fn += ".fpz"

        array_dict = {
            "data": self.array.data,
            "shape": self.array.shape,
            "indices": self.array.indices,
            "indptr": self.array.indptr,
            "fp_names": np.array(self.fp_names),
            "level": self.level,
            "name": self.name,
            "fp_type": self.fp_type,
        }

        for k, v in self.props.items():
            array_dict["_" + str(k)] = v

        with open(fn, "wb") as f:
            np.savez_compressed(f, **array_dict)

    def savetxt(self, fn, with_names=True):
        """Save bitstring representation to text file.

        Only implemented for `fp_type` of `Fingerprint`. This should not be
        attempted for large numbers of bits.

        Parameters
        ----------
        fn : str or filehandle
            Out file. Extension is automatically parsed to determine whether
            compression is used.
        with_names : bool, optional
            Include name of fingerprint in same row after bitstring.

        Raises
        ------
        E3FPInvalidFingerprintError
            If `fp_type` is not `Fingerprint`.
        E3FPEfficiencyWarning
            If `bits` is over 2^14 = 16384.
        """
        if self.fp_type is not Fingerprint:
            raise E3FPInvalidFingerprintError(
                "Only binary `Fingerprint` databases may be saved to "
                "bitstrings."
            )

        if self.bits > 2 ** 14:
            warnings.warn(
                (
                    "Saving sparse bitstrings to text file is highly "
                    "inefficient for large bit lengths"
                ),
                category=E3FPEfficiencyWarning,
                stacklevel=2,
            )

        row_fmt = "{0:s}"
        if with_names:
            row_fmt += " {1:s}"

        with smart_open(fn, "w") as f:
            for i in range(self.fp_num):
                # Much more efficient to access underlying arrays
                indices = self.array.indices[
                    self.array.indptr[i] : self.array.indptr[i + 1]
                ]
                bs = "1".join(
                    [
                        "0" * j
                        for j in np.diff(np.r_[-1, indices, self.bits]) - 1
                    ]
                )
                f.write(row_fmt.format(bs, self.fp_names[i]) + "\n")

    @classmethod
    def load(cls, fn):
        """Load database from file.

        The extension is used to determine how database was serialized
        (`save` vs `savez`).

        Parameters
        ----------
        fn : str
            Filename

        Returns
        -------
        FingerprintDatabase
            Database
        """
        if fn.endswith(".fpz"):
            if scipy.__version__ < "1.0":
                warnings.warn(
                    (
                        "Use SciPy 1.0 or newer to efficiently load large "
                        "FingerprintDatabases."
                    ),
                    category=E3FPEfficiencyWarning,
                    stacklevel=2,
                )
            array_dict = dict(np.load(fn, allow_pickle=True).items())
            props_dict = {}
            for k in list(array_dict.keys()):
                if k.startswith("_"):
                    v = array_dict.pop(k)
                    props_dict[k[1:]] = v
            array = csr_matrix(
                (
                    array_dict["data"],
                    array_dict["indices"],
                    array_dict["indptr"],
                ),
                shape=array_dict["shape"],
            )
            return FingerprintDatabase.from_array(
                array,
                array_dict["fp_names"],
                fp_type=array_dict["fp_type"].item(),
                level=array_dict["level"].item(),
                name=array_dict["name"].item(),
                props=props_dict,
            )
        else:
            with smart_open(fn) as f:
                return pkl.load(f)

    @property
    def fp_num(self):
        try:
            return self.array.shape[0]
        except AttributeError:
            return 0

    @property
    def bits(self):
        try:
            return self.array.shape[1]
        except AttributeError:
            return None

    def get_prop(self, key):
        """Get property.

        Raises
        ------
        KeyError
            If `key` not in `props`.
        """
        try:
            return self.props[key]
        except KeyError:
            raise KeyError("Database does not have property.")

    def set_prop(self, key, vals, check_length=True):
        """Set values of property for fingerprints.

        Parameters
        ----------
        key : str
            Name of property
        vals : array_like
            Values of property.
        check_length : bool, optional
            Check to ensure number of properties match number of fingerprints
            already in database. This should only be set to False for
            temporary iterative updating.
        """
        vals = np.asanyarray(vals)
        if check_length and vals.shape[0] != len(self.fp_names):
            raise ValueError("props must have the same count as fingerprints.")
        self.props[key] = vals

    def _get_fprint_at_index(self, i):
        return self.fp_type.from_vector(
            self.array[i, :],
            level=self.level,
            name=self.fp_names[i],
            props=self._get_fprint_props(i),
        )

    def _get_fprint_props(self, i):
        return {k: v[i] for k, v in self.props.items()}

    def _check_fingerprints_are_valid(self, fprints):
        """Check if passed fingerprints fit database."""
        if fprints[0].level != self.level:
            raise ValueError(
                "Provided fingerprints must have database level"
                " {}".format(self.level)
            )
        if self.fp_type is None:
            self.fp_type = fprints[0].__class__
        elif self.fp_type is not fprints[0].__class__:
            logging.warning(
                "Database is of type {}. Fingerprints will be cast"
                " to this type.".format(self.fp_type.__name__)
            )

    def __eq__(self, other):
        if (
            self.fp_type == other.fp_type
            and self.level == other.level
            and self.bits == other.bits
            and self.fp_num == other.fp_num
            and self.fp_names_to_indices == other.fp_names_to_indices
        ):
            if self.array is None or other.array is None:
                return self.array is other.array
            else:
                return (self.array - other.array).nnz == 0
        else:
            return False

    def __neq__(self, other):
        return not self == other

    def __iter__(self):
        for i in range(self.fp_num):
            yield self.fp_type.from_vector(
                self.array[i, :], level=self.level, name=self.fp_names[i]
            )

    def __add__(self, other):
        return concat([self, other])

    def __repr__(self):
        return "FingerprintDatabase(fp_type={}, level={}, name='{}')".format(
            self.fp_type.__name__, self.level, self.name
        )

    def __str__(self):
        return (
            "FingerprintDatabase[name: {}, fp_type: {}, level: {}, "
            "bits: {}, fp_num: {}]"
        ).format(
            self.name,
            self.fp_type.__name__,
            self.level,
            self.bits,
            self.fp_num,
        )

    def __len__(self):
        return self.fp_num

    def __getitem__(self, key):
        """Get list of fingerprints with name."""
        if isinstance(key, str):
            try:
                indices = self.fp_names_to_indices[key]
            except AttributeError:
                raise KeyError(
                    "fingerprint named {} is not in the database".format(key)
                )
            return [self._get_fprint_at_index(i) for i in indices]
        elif isinstance(key, int):
            try:
                return self._get_fprint_at_index(key)
            except (IndexError, AttributeError):
                raise IndexError("index out of range")
        else:
            raise TypeError("Key or index must be str or int.")

    def __copy__(self):
        return FingerprintDatabase.from_array(
            self.array,
            self.fp_names,
            fp_type=self.fp_type,
            level=self.level,
            name=self.name,
            props=self.props,
        )

    def __getstate__(self):
        d = {}
        d["name"] = self.name
        d["fp_type"] = self.fp_type
        d["level"] = self.level
        d["array"] = self.array
        d["fp_names"] = self.fp_names
        d["props"] = self.props
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__["fp_names_to_indices"] = defaultdict(list)
        self.update_names_map()
        if "props" not in state:
            self.props = {}


@deprecated("1.2", msg="Use `concat` instead.")
def append(dbs):
    """Efficiently concatenate `FingerprintDatabase` objects.

    The databases must be of the same type with the same number of bits,
    level, and property names.

    Parameters
    ----------
    dbs : iterable of FingerprintDatabase
        Fingerprint databases

    Returns
    -------
    FingerprintDatabase
        Database with all fingerprints from provided databases.
    """
    return concat(dbs)


def concat(dbs):
    """Efficiently concatenate `FingerprintDatabase` objects.

    The databases must be of the same type with the same number of bits,
    level, and property names.

    Parameters
    ----------
    dbs : iterable of FingerprintDatabase
        Fingerprint databases

    Returns
    -------
    FingerprintDatabase
        Database with all fingerprints from provided databases.

    See Also
    --------
    FingerprintDatabase

    Examples
    --------
    >>> from e3fp.fingerprint.db import FingerprintDatabase, concat
    >>> from e3fp.fingerprint.fprint import Fingerprint
    >>> import numpy as np
    >>> np.random.seed(2)
    >>> db1 = FingerprintDatabase(fp_type=Fingerprint, name="TestDB1", level=5)
    >>> db2 = FingerprintDatabase(fp_type=Fingerprint, name="TestDB2", level=5)
    >>> bvs = (np.random.uniform(size=(6, 1024)) > .9).astype(bool)
    >>> fps = [Fingerprint.from_vector(bvs[i, :], name="fp" + str(i), level=5)
    ...        for i in range(bvs.shape[0])]
    >>> db1.add_fingerprints(fps[:3])
    >>> db2.add_fingerprints(fps[3:])
    >>> print(concat([db1, db2]))
    FingerprintDatabase[name: None, fp_type: Fingerprint, level: 5, bits: 1024, fp_num: 6]
    """
    dbs = list(dbs)
    level = dbs[0].level
    bits = dbs[0].bits
    fp_type = dbs[0].fp_type
    arrays = []
    fp_names = []
    full_db = FingerprintDatabase(fp_type=fp_type, level=level)
    for i, db in enumerate(dbs):
        if db.level != level:
            raise TypeError(
                "Cannot concatenate databases with different levels"
            )
        elif db.bits != bits:
            raise TypeError(
                "Cannot concatenate databases with different bit lengths"
            )
        elif db.fp_type != fp_type:
            raise TypeError(
                "Cannot concatenate databases with different "
                "fingerprint types"
            )
        arrays.append(db.array)
        fp_names.extend(db.fp_names)
        full_db.update_props(db.props, append=True, check_length=False)

    full_db.array = vstack(arrays).tocsr()
    full_db.fp_names = fp_names

    for prop_name, prop_vals in full_db.props.items():
        if len(prop_vals) != full_db.fp_num:
            raise ValueError("props must have the same count as fingerprints.")

    full_db.update_names_map()
    return full_db

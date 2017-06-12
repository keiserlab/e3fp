"""Database for storing and serializing fingerprints.

Author: Seth Axen
E-mail: seth.axen@gmail.com"""
from collections import defaultdict
try:
    import cPickle as pkl
except ImportError:  # Python 3
    import pickle as pkl
import logging

from scipy.sparse import vstack
from python_utilities.io_tools import smart_open
from .fprint import Fingerprint, CountFingerprint, FloatFingerprint


class FingerprintDatabase(object):

    """A database for storing, saving, and loading fingerprints.

    Fingerprints must have the same bit length and be of the same level.
    Additionally, they are all be cast to the type of fingerprint passed
    to the database upon instantiation.

    Attributes
    ----------
    array : csr_matrix
        Sparse matrix with dimensions N x M, where M is the number
        of bits in the fingerprints.
    fp_names : list of str
        Names of fingerprints
    fp_names_to_indices : dict
        Map from fingerprint name to row indices of `array`
    fp_type : type
        Type of fingerprint (Fingerprint, CountsFingerprint, FloatFingerprint)
    fp_num : int
        Number of fingerprints in database
    bits : int
        Number of bits of fingerprints
    level : int
        Level, or number of iterations used during fingerprinting.
    name : str
        Name of database
    """

    def __init__(self, fp_type=Fingerprint, level=-1, name=None):
        """Constructor

        Parameters
        ----------
        fp_type : type, optional
            Type of fingerprint (Fingerprint, CountsFingerprint,
            FloatFingerprint)
        level : int, optional
            Level, or number of iterations used during fingerprinting.
        name : str, optional
            Name of database
        """
        if fp_type not in (Fingerprint, CountFingerprint,
                           FloatFingerprint):
            raise TypeError(
                "{} is not a valid fingerprint type".format(fp_type))
        self.name = name
        self.fp_type = fp_type
        self.level = level
        self.array = None
        self.fp_names = []
        self.fp_names_to_indices = defaultdict(list)

    def add_fingerprints(self, fprints):
        """Add fingerprints to database.

        Parameters
        ----------
        fprints : iterable of Fingerprint
            Fingerprints to add to database
        """
        self._check_fingerprints_are_valid(fprints)

        dtype = self.fp_type.vector_dtype

        new_rows, new_names = list(zip(*[
            (fprint.to_vector(sparse=True, dtype=dtype), fprint.name)
            for fprint in fprints]))
        try:
            old_fp_num = self.array.shape[0]
            self.array = vstack([self.array] + list(new_rows))
        except (AttributeError, ValueError):  # array not yet defined
            old_fp_num = 0
            self.array = vstack(new_rows)
        self.array = self.array.tocsr()
        del new_rows
        self.fp_names += new_names
        self.update_names_map(new_names=new_names, offset=old_fp_num)

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

    @classmethod
    def load(cls, fn):
        """Load database from file.

        Parameters
        ----------
        fn : str
            Filename

        Returns
        -------
        FingerprintDatabase
            Dabatase
        """
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

    def _check_fingerprints_are_valid(self, fprints):
        """Check if passed fingerprints fit database."""
        if fprints[0].level != self.level:
            raise ValueError("Provided fingerprints must have database level"
                             " {}".format(self.level))
        if self.fp_type is None:
            self.fp_type = fprints[0].__class__
        elif self.fp_type is not fprints[0].__class__:
            logging.warning("Database is of type {}. Fingerprints will be cast"
                            " to this type.")

    def __iter__(self):
        for i in range(self.fp_num):
            yield self.fp_type.from_vector(self.array[i, :], level=self.level,
                                           name=self.fp_names[i])

    def __add__(self, other):
        if self.level != other.level:
            raise TypeError("Cannot add databases with different levels")
        elif self.bits != other.bits:
            raise TypeError("Cannot add databases with different bit lengths")
        elif self.fp_type != other.fp_type:
            raise TypeError(
                "Cannot add databases with different fingerprint types")
        db = FingerprintDatabase(fp_type=self.fp_type, level=self.level)
        db.array = vstack([self.array, other.array]).tocsr()
        db.fp_names = self.fp_names + other.fp_names
        db.update_names_map()
        return db

    def __repr__(self):
        return "FingerprintDatabase(fp_type={}, level={}, name={})".format(
            self.fp_type, self.level, self.name)

    def __str__(self):
        return ("FingerprintDatabase[name: {}  fp_type: {}  level: {}"
                "  bits: {}  fp_num: {}]").format(self.name,
                                                  self.fp_type.__name__,
                                                  self.level, self.bits,
                                                  self.fp_num)

    def __len__(self):
        return self.fp_num

    def __getitem__(self, key):
        """Get list of fingerprints with name."""
        if isinstance(key, str):
            try:
                array = self.array[self.fp_names_to_indices[key]]
            except AttributeError:
                raise KeyError(
                    "fingerprint named {} is not in the database".format(key))
            return [self.fp_type.from_vector(array[i, :], level=self.level,
                                             name=key)
                    for i in range(array.shape[0])]
        elif isinstance(key, int):
            try:
                return self.fp_type.from_vector(self.array[key, :],
                                                level=self.level,
                                                name=self.fp_names[key])
            except (IndexError, AttributeError):
                raise IndexError("index out of range")
        else:
            raise TypeError("Key or index must be str or int.")

    def __getstate__(self):
        d = {}
        d["name"] = self.name
        d["fp_type"] = self.fp_type
        d["level"] = self.level
        d["array"] = self.array
        d["fp_names"] = self.fp_names
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__["fp_names_to_indices"] = defaultdict(list)
        self.update_names_map()

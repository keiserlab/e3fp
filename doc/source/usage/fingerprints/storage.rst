Fingerprint Storage
===================

The most efficient way to store and interact with fingerprints is through the
`e3fp.fingerprint.db.FingerprintDatabase` class. This class wraps a matrix with
sparse rows (`scipy.sparse.csr_matrix`), where each row is a fingerprint. This
enables rapid I/O of the database while also minimizing the memory footprint.
Accessing the underlying sparse representation with the
:ref:`.FingerprintDatabase.array` attribute is convenient for machine learning
purposes, while the database class itself provides several useful functions.

.. note::

    We strongly recommend upgrading to at least SciPy v1.0.0 when working with
    large fingerprint databases, as old versions are much slower and have
    several bugs for database loading.


Database I/O and Indexing
-------------------------

See the full `e3fp.fingerprint.db.FingerprintDatabase` documentation for a
description of basic database usage, attributes, and methods. Below, several
additional use cases are documented.

Batch Database Operations
-------------------------

Due to the sparse representation of the underlying data structure, an un-
folded database, a database with unfolded fingerprints does not use
significantly more disk space than a database with folded fingerprints. However,
it is usually necessary to fold fingerprints for machine learning tasks. The
:py:class:`.FingerprintDatabase` does this very quickly.

.. testsetup::

    import numpy as np
    np.random.seed(3)

.. doctest::

    >>> from e3fp.fingerprint.db import FingerprintDatabase
    >>> from e3fp.fingerprint.fprint import Fingerprint
    >>> import numpy as np
    >>> db = FingerprintDatabase(fp_type=Fingerprint, name="TestDB")
    >>> print(db)
    FingerprintDatabase[name: TestDB, fp_type: Fingerprint, level: -1, bits: None, fp_num: 0]
    >>> on_inds = [np.random.uniform(0, 2**32, size=30) for i in range(5)]
    >>> fps = [Fingerprint(x, bits=2**32) for x in on_inds]
    >>> db.add_fingerprints(fps)
    >>> print(db)
    FingerprintDatabase[name: TestDB, fp_type: Fingerprint, level: -1, bits: 4294967296, fp_num: 5]
    >>> db.get_density()
    6.984919309616089e-09
    >>> fold_db = db.fold(1024)
    >>> print(fold_db)
    FingerprintDatabase[name: TestDB, fp_type: Fingerprint, level: -1, bits: 1024, fp_num: 5]
    >>> fold_db.get_density()
    0.0287109375

A database can be converted to a different fingerprint type:

    >>> from e3fp.fingerprint.fprint import CountFingerprint 
    >>> count_db = db.as_type(CountFingerprint)
    >>> print(count_db)
    FingerprintDatabase[name: TestDB, fp_type: CountFingerprint, level: -1, bits: 4294967296, fp_num: 5]
    >>> count_db[0]
    CountFingerprint(counts={2977004690: 1, ..., 3041471738: 1}, level=-1, bits=4294967296, name=None)

The `e3fp.fingerprint.db.concat` method allows efficient joining of multiple
databases.

    >>> from e3fp.fingerprint.db import concat
    >>> dbs = []
    >>> for i in range(10):
    ...     db = FingerprintDatabase(fp_type=Fingerprint)
    ...     on_inds = [np.random.uniform(0, 1024, size=30) for j in range(5)]
    ...     fps = [Fingerprint(x, bits=2**32, name="Mol{}".format(i)) for x in on_inds]
    ...     db.add_fingerprints(fps)
    ...     dbs.append(db)
    >>> dbs[0][0]
    Fingerprint(indices=array([94, 97, ..., 988, 994]), level=-1, bits=4294967296, name=Mol0)
    >>> print(dbs[0])
    FingerprintDatabase[name: None, fp_type: Fingerprint, level: -1, bits: 4294967296, fp_num: 5]
    >>> merge_db = concat(dbs)
    >>> print(merge_db)
    FingerprintDatabase[name: None, fp_type: Fingerprint, level: -1, bits: 4294967296, fp_num: 50]

Database Comparison
-------------------

Two databases may be compared using various metrics in
`e3fp.fingerprint.metrics`. Additionally, all fingerprints in a database may be
compared to each other simply by only providing a single database.
See :ref:`Fingerprint Comparison` for more details.

Performing Machine Learning on the Database
-------------------------------------------

The underlying sparse matrix may be passed directly to machine learning tools
in any package that is compatible with SciPy sparse matrices, such as
`scikit-learn <http://scikit-learn.org/>`_.

    >>> from sklearn.naive_bayes import BernoulliNB
    >>> clf = BernoulliNB()
    >>> clf.fit(db.array, ypred)  # doctest: +SKIP
    BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
    >>> clf.predict(db2.array)   # doctest: +SKIP
    ...

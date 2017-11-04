Fingerprint Comparison
======================

The `e3fp.fingerprint.metrics` sub-package provides several useful methods for
batch comparison of fingerprints in various representations.

Fingerprint Metrics
-------------------

These metrics operate directly on pairs of :py:class:`.Fingerprint` and
:py:class:`.FingerprintDatabase` objects or on a combination of each. If
only a single variable is specified, self-comparison is performed. The
implemented methods are common functions for fingerprint similarity in the
literature.

.. todo::

    Document examples

Array Metrics
-------------

To efficiently compare fingerprint databases above, we provide comparison
metrics that can operate directly on the internal sparse matrix representation
without the need to "densify it". We describe these here, as they have several
additional features.

The array metrics implemented in `e3fp.fingerprint.metrics.array_metrics` are
implemented such that they may take any combination of dense and sparse inputs.
Additionally, they are designed to function as
`scikit-learn-compatible kernels <http://scikit-learn.org/stable/modules/metrics.html>`_
for machine learning tasks. For example, one might perform an analysis using a
support vector machine (SVM) and Tanimoto kernel.

.. code:: python
    
    >>> from sklearn.svm import SVC
    >>> from e3fp.fingerprint.metrics.array_metrics import tanimoto
    >>> clf = SVC(kernel=tanimoto)
    >>> clf.fit(X, y)
    ...
    >>> clf.predict(test)
    ...

Most common fingerprint comparison metrics only apply to binary fingerprints.
We include several that operate equally well on count- and float-based
fingerprints. For example, to our knowledge, we provide the only open source
implementation of Soergel similarity, the analog to the Tanimoto coefficient
for non-binary fingerprints that can efficiently operate on sparse inputs.

.. code:: python
    
    >>> from e3fp.fingerprint.metrics.array_metrics import soergel
    >>> clf = SVC(kernel=soergel)
    >>> clf.fit(X, y)
    ...
    >>> clf.predict(test)
    ...

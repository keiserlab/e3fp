Fingerprints
============

The simplest interface for molecular fingerprints are through three classes in
`e3fp.fingerprint.fprint`:

:py:class:`.Fingerprint`
    a fingerprint with "on" bits

:py:class:`.CountFingerprint`
    a fingerprint with counts for each "on" bit

:py:class:`.FloatFingerprint`
    a fingerprint with float values for each "on" bit, generated for example by
    averaging conformer fingerprints.

In addition to storing "on" indices and, for the latter two, corresponding
values, they store fingerprint properties, such as name, level, and any
arbitrary property. They also provide simple interfaces for fingerprint
comparison, some basic processing, and comparison.

.. note:: Many of these operations are more efficient when operating on a
    :py:class:`.FingerprintDatabase`. See :ref:`Fingerprint Storage` for more
    information.

In the below examples, we will focus on :py:class:`.Fingerprint` and
:py:class:`.CountFingerprint`. First, we execute the necessary imports.

.. testsetup::

    import numpy as np
    np.random.seed(0)

.. doctest::

    >>> from e3fp.fingerprint.fprint import Fingerprint, CountFingerprint
    >>> import numpy as np

.. seealso::

    :ref:`Fingerprint Storage`, :ref:`Fingerprint Comparison`

Creation and Conversion
-----------------------

Here we create a bit-fingerprint with random "on" indices.

    >>> bits = 2**32
    >>> indices = np.sort(np.random.randint(0, bits, 30))
    >>> indices
    array([ 243580376,  305097549, ..., 3975407269, 4138900056])
    >>> fp1 = Fingerprint(indices, bits=bits, level=0)
    >>> fp1
    Fingerprint(indices=array([243580376, ..., 4138900056]), level=0, bits=4294967296, name=None)

This fingerprint is extremely sparse

    >>> fp1.bit_count
    30
    >>> fp1.density
    6.984919309616089e-09

We can therefore "fold" the fingerprint through a series of bitwise "OR"
operations on halves of the sparse vector until it is of a specified length,
with minimal collision of bits.

    >>> fp_folded = fp1.fold(1024)
    >>> fp_folded
    Fingerprint(indices=array([9, 70, ..., 845, 849]), level=0, bits=1024, name=None)
    >>> fp_folded.bit_count
    29
    >>> fp_folded.density
    0.0283203125

A :py:class:`.CountFingerprint` may be created by also providing a dictionary
matching indices with nonzero counts to the counts.

    >>> indices2 = np.sort(np.random.randint(0, bits, 60))
    >>> counts = dict(zip(indices2, np.random.randint(1, 10, indices2.size)))
    >>> counts
    {80701568: 8, 580757632: 7, ..., 800291326: 5, 4057322111: 7}
    >>> cfp1 = CountFingerprint(counts=counts, bits=bits, level=0)
    >>> cfp1
    CountFingerprint(counts={80701568: 8, 580757632: 7, ..., 3342157822: 2, 4057322111: 7}, level=0, bits=4294967296, name=None)

Unlike folding a bit fingerprint, by default, folding a count fingerprint
performs a "SUM" operation on colliding counts.

    >>> cfp1.bit_count
    60
    >>> cfp_folded = cfp1.fold(1024)
    >>> cfp_folded
    CountFingerprint(counts={128: 15, 257: 4, ..., 1022: 2, 639: 7}, level=0, bits=1024, name=None)
    >>> cfp_folded.bit_count
    57

It is trivial to interconvert the fingerprints.

    >>> cfp_folded2 = CountFingerprint.from_fingerprint(fp_folded)
    >>> cfp_folded2
    CountFingerprint(counts={9: 1, 87: 1, ..., 629: 1, 763: 1}, level=0, bits=1024, name=None)
    >>> cfp_folded2.indices[:5]
    array([  9,  70,  72,  87, 174])
    >>> fp_folded.indices[:5]
    array([  9,  70,  72,  87, 174])

RDKit Morgan fingerprints (analogous to ECFP) may easily be converted to a
:py:class:`.Fingerprint`.

    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem
    >>> mol = Chem.MolFromSmiles('Cc1ccccc1')
    >>> mfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    >>> mfp
    <rdkit.DataStructs.cDataStructs.ExplicitBitVect object at 0x...>
    >>> Fingerprint.from_rdkit(mfp)
    Fingerprint(indices=array([389, 1055, ..., 1873, 1920]), level=-1, bits=2048, name=None)

Likewise, :py:class:`.Fingerprint` can be easily converted to a NumPy ndarray or
SciPy sparse matrix.

    >>> fp_folded.to_vector()
    <1x1024 sparse matrix of type '<type 'numpy.bool_'>'
    ...with 29 stored elements in Compressed Sparse Row format>
    >>> fp_folded.to_vector(sparse=False)
    array([False, False, False, ..., False, False, False], dtype=bool)
    >>> np.where(fp_folded.to_vector(sparse=False))[0]
    array([  9,  70,  72,  87, ...])
    >>> cfp_folded.to_vector(sparse=False)
    array([0, 0, 0, ..., 0, 2, 0], dtype=uint16)
    >>> cfp_folded.to_vector(sparse=False).sum()
    252

Algebra
-------

Basic algebraic functions may be performed on fingerprints. If either
fingerprint is a bit fingerprint, all algebraic functions are bit-wise.
The following bit-wise operations are supported:

Equality
    >>> fp1 = Fingerprint([0, 1, 6, 8, 12], bits=16)
    >>> fp2 = Fingerprint([1, 2, 4, 8, 11, 12], bits=16)
    >>> fp1 == fp2
    False
    >>> fp1_copy = Fingerprint.from_fingerprint(fp1)
    >>> fp1 == fp1_copy
    True
    >>> fp1_copy.level = 5
    >>> fp1 == fp1_copy
    False

Union/OR
    >>> fp1 + fp2
    Fingerprint(indices=array([0, 1, 2, 4, 6, 8, 11, 12]), level=-1, bits=16, name=None)
    >>> fp1 | fp2
    Fingerprint(indices=array([0, 1, 2, 4, 6, 8, 11, 12]), level=-1, bits=16, name=None)

Intersection/AND
    >>> fp1 & fp2
    Fingerprint(indices=array([1, 8, 12]), level=-1, bits=16, name=None)

Difference/AND NOT
    >>> fp1 - fp2
    Fingerprint(indices=array([0, 6]), level=-1, bits=16, name=None)
    >>> fp2 - fp1
    Fingerprint(indices=array([2, 4, 11]), level=-1, bits=16, name=None)

XOR
    >>> fp1 ^ fp2
    Fingerprint(indices=array([0, 2, 4, 6, 11]), level=-1, bits=16, name=None)

With count or float fingerprints, bit-wise operations are still possible, but
algebraic operations are applied to counts.

    >>> fp1 = CountFingerprint(counts={0: 3, 1: 2, 5: 1, 9: 3}, bits=16)
    >>> fp2 = CountFingerprint(counts={1: 2, 5: 2, 7: 3, 10: 7}, bits=16)
    >>> fp1 + fp2
    CountFingerprint(counts={0: 3, 1: 4, 5: 3, 7: 3, 9: 3, 10: 7}, level=-1, bits=16, name=None)
    >>> fp1 - fp2
    CountFingerprint(counts={0: 3, 1: 0, 5: -1, 7: -3, 9: 3, 10: -7}, level=-1, bits=16, name=None)
    >>> fp1 * 3
    CountFingerprint(counts={0: 9, 1: 6, 5: 3, 9: 9}, level=-1, bits=16, name=None)
    >>> fp1 / 2
    FloatFingerprint(counts={0: 1.5, 1: 1.0, 5: 0.5, 9: 1.5}, level=-1, bits=16, name=None)

Finally, fingerprints may be batch added and averaged, producing either a count
or float fingerprint when sensible.

    >>> from e3fp.fingerprint.fprint import add, mean
    >>> fps = [Fingerprint(np.random.randint(0, 32, 8), bits=32) for i in range(100)]
    >>> add(fps)
    CountFingerprint(counts={0: 23, 1: 23, ..., 30: 20, 31: 14}, level=-1, bits=32, name=None)
    >>> mean(fps)
    FloatFingerprint(counts={0: 0.23, 1: 0.23, ..., 30: 0.2, 31: 0.14}, level=-1, bits=32, name=None)

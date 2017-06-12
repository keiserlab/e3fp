"""Tests for fingerprint databases.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import unittest
import tempfile

import numpy as np


class FingerprintDatabaseTestCases(unittest.TestCase):

    """Tests for FingerprintDatabase"""

    def test_create(self):
        from e3fp.fingerprint.fprint import CountFingerprint
        from e3fp.fingerprint.db import FingerprintDatabase
        db = FingerprintDatabase(fp_type=CountFingerprint, level=5,
                                 name="TestDB")
        self.assertIs(db.fp_type, CountFingerprint)
        self.assertEqual(db.level, 5)
        self.assertEqual(db.name, "TestDB")

    def test_db_equality(self):
        from e3fp.fingerprint.fprint import Fingerprint, CountFingerprint
        from e3fp.fingerprint.db import FingerprintDatabase
        db1 = FingerprintDatabase(name="TestDB")
        db2 = FingerprintDatabase(name="TestDB2")
        self.assertEqual(db1, db2)
        db2 = FingerprintDatabase(fp_type=CountFingerprint, name="TestDB2")
        self.assertNotEqual(db1, db2)
        db2 = FingerprintDatabase(level=5, name="TestDB2")
        self.assertNotEqual(db1, db2)
        db2 = FingerprintDatabase(name="TestDB2")
        array = (
            np.random.uniform(0, 1, size=(10, 1024)) > .9).astype(np.uint16)
        fprints = [Fingerprint.from_vector(array[i, :])
                   for i in range(10)]
        for i, fp in enumerate(fprints):
            name = str(i)
            fp.name = name
        db1.add_fingerprints(fprints)
        db2.add_fingerprints(fprints)
        self.assertEqual(db1, db2)
        db2.add_fingerprints([fprints[0]])
        self.assertNotEqual(db1, db2)

    def test_create_from_array(self):
        from e3fp.fingerprint.fprint import Fingerprint, CountFingerprint
        from e3fp.fingerprint.db import FingerprintDatabase
        array = (
            np.random.uniform(0, 1, size=(10, 1024)) > .9).astype(np.uint16)
        fprints = [Fingerprint.from_vector(array[i, :])
                   for i in range(10)]
        fp_names = []
        for i, fp in enumerate(fprints):
            name = str(i)
            fp.name = name
            fp.level = 5
            fp_names.append(name)
        db1 = FingerprintDatabase(fp_type=CountFingerprint, level=5,
                                  name="Test")
        db1.add_fingerprints(fprints)
        db2 = FingerprintDatabase.from_array(array, fp_names, level=5,
                                             name="Test")
        self.assertEqual(db1.fp_type, db2.fp_type)
        np.testing.assert_array_equal(db1.array.todense().getA(),
                                      db2.array.todense().getA())

    def test_add_fingerprints(self):
        from e3fp.fingerprint.fprint import CountFingerprint
        from e3fp.fingerprint.db import FingerprintDatabase
        array = (
            np.random.uniform(0, 1, size=(10, 1024)) > .9).astype(np.double)
        fprints = [CountFingerprint.from_vector(array[i, :])
                   for i in range(10)]
        for i, fp in enumerate(fprints):
            fp.name = str(i)
        db = FingerprintDatabase(fp_type=CountFingerprint)
        db.add_fingerprints(fprints)
        self.assertIs(db.fp_type, CountFingerprint)
        self.assertTrue(np.issubdtype(db.array.dtype, np.uint16))
        self.assertEqual(db.fp_num, 10)
        self.assertEqual(db.bits, 1024)
        self.assertEqual(len(db.fp_names_to_indices), 10)
        for k, v in db.fp_names_to_indices.items():
            k = int(k)
            self.assertEqual(k, v[0])
            np.testing.assert_almost_equal(
                array[k, :], db.array[k, :].todense().getA().flatten())

    def test_add_fingerprints_two_batches(self):
        from e3fp.fingerprint.fprint import CountFingerprint
        from e3fp.fingerprint.db import FingerprintDatabase
        array = (
            np.random.uniform(0, 1, size=(10, 1024)) > .9).astype(np.double)
        fprints = [CountFingerprint.from_vector(array[i, :])
                   for i in range(10)]
        for i, fp in enumerate(fprints):
            fp.name = str(i)
        db = FingerprintDatabase(fp_type=CountFingerprint)
        db.add_fingerprints(fprints)
        fprints2 = [CountFingerprint.from_vector(array[i, :])
                    for i in range(10)]
        for i, fp in enumerate(fprints2):
            fp.name = str(i + len(fprints))
        db.add_fingerprints(fprints2)

        self.assertIs(db.fp_type, CountFingerprint)
        self.assertTrue(np.issubdtype(db.array.dtype, np.uint16))
        self.assertEqual(db.fp_num, 20)
        self.assertEqual(db.bits, 1024)
        self.assertEqual(len(db.fp_names_to_indices), 20)
        for k, v in db.fp_names_to_indices.items():
            k = int(k)
            self.assertEqual(k, v[0])
            np.testing.assert_almost_equal(
                array[k % 10, :], db.array[k, :].todense().getA().flatten())

    def test_duplicate_fprint_names_detected(self):
        from e3fp.fingerprint.fprint import CountFingerprint
        from e3fp.fingerprint.db import FingerprintDatabase
        array = (
            np.random.uniform(0, 1, size=(10, 1024)) > .9).astype(np.double)
        fprints = [CountFingerprint.from_vector(array[i, :])
                   for i in range(10)]
        for i, fp in enumerate(fprints):
            fp.name = str(i)
        db = FingerprintDatabase(fp_type=CountFingerprint)
        db.add_fingerprints(fprints)
        fprints2 = [CountFingerprint.from_vector(array[i, :])
                    for i in range(10)]
        for i, fp in enumerate(fprints2):
            fp.name = str(i)
        db.add_fingerprints(fprints2)

        self.assertIs(db.fp_type, CountFingerprint)
        self.assertTrue(np.issubdtype(db.array.dtype, np.uint16))
        self.assertEqual(db.fp_num, 20)
        self.assertEqual(db.bits, 1024)
        self.assertEqual(len(db.fp_names_to_indices), 10)
        for k, v in db.fp_names_to_indices.items():
            k = int(k)
            self.assertEqual(k, v[0])
            self.assertEqual(k, v[1] - 10)
            np.testing.assert_almost_equal(
                array[k % 10, :], db.array[k, :].todense().getA().flatten())

    def test_get_db_subset(self):
        from e3fp.fingerprint.db import FingerprintDatabase
        array = (
            np.random.uniform(0, 1, size=(10, 1024)) > .9).astype(np.uint16)
        fp_names = []
        for i in range(array.shape[0]):
            fp_names.append(str(i))
        db = FingerprintDatabase.from_array(array, fp_names=fp_names, level=5)
        sub_db = db.get_subset(fp_names[:-2])
        self.assertEqual(db.level, sub_db.level)
        self.assertEqual(db.fp_type, sub_db.fp_type)
        self.assertEqual((db.array[:-2, :] - sub_db.array).nnz, 0)

    def test_roundtrip(self):
        """Ensure DB is the same after saving and loading."""
        from e3fp.fingerprint.db import FingerprintDatabase
        array = (
            np.random.uniform(0, 1, size=(10, 1024)) > .9).astype(np.uint16)
        fp_names = []
        for i in range(array.shape[0]):
            fp_names.append(str(i))
        db = FingerprintDatabase.from_array(array, fp_names=fp_names, level=5)
        desc, db_file = tempfile.mkstemp(suffix=".fps.bz2")
        os.close(desc)
        db.save(db_file)
        db2 = db.load(db_file)
        os.unlink(db_file)
        self.assertEqual(db, db2)


if __name__ == "__main__":
    unittest.main()

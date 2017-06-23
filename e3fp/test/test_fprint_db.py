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
        db = FingerprintDatabase.from_array(array, fp_names=fp_names, level=5,
                                            props={"index": range(10)})
        desc, db_file = tempfile.mkstemp(suffix=".fps.bz2")
        os.close(desc)
        db.save(db_file)
        db2 = db.load(db_file)
        os.unlink(db_file)
        self.assertEqual(db, db2)
        self.assertListEqual(db2.get_prop("index").tolist(), list(range(10)))

    def test_roundtrip2(self):
        """Ensure DB is the same after saving and loading."""
        from e3fp.fingerprint.fprint import Fingerprint
        from e3fp.fingerprint.db import FingerprintDatabase
        fprints = []
        for i in range(10):
            fprints.append(
                Fingerprint.from_indices(np.random.uniform(0, 2**32, size=30),
                                         bits=2**32, level=5))
            fprints[-1].name = "fp" + str(i)
            fprints[-1].set_prop("index", float(i))
        db = FingerprintDatabase(fp_type=Fingerprint, level=5)
        db.add_fingerprints(fprints)
        desc, db_file = tempfile.mkstemp(suffix=".fps.bz2")
        os.close(desc)
        db.save(db_file)
        db2 = db.load(db_file)
        os.unlink(db_file)
        self.assertEqual(db, db2)
        self.assertListEqual(db2.get_prop("index").tolist(), list(range(10)))

    def test_lookup(self):
        from e3fp.fingerprint.fprint import Fingerprint
        from e3fp.fingerprint.db import FingerprintDatabase
        db = FingerprintDatabase(name="TestDB")
        array = (
            np.random.uniform(0, 1, size=(2, 1024)) > .9).astype(np.bool_)
        fp_names = []
        for i in range(array.shape[0]):
            fp_names.append("fp" + str(i))
        db = FingerprintDatabase.from_array(array, fp_names, name="Test")
        for i in range(array.shape[0]):
            self.assertEqual(Fingerprint.from_vector(array[i, :]), db[i])
            self.assertEqual(Fingerprint.from_vector(array[i, :]),
                             db[db.fp_names[i]][0])

    def test_fold_db(self):
        from e3fp.fingerprint.fprint import Fingerprint, CountFingerprint
        from e3fp.fingerprint.db import FingerprintDatabase
        fold_len = 32
        for fp_type in (Fingerprint, CountFingerprint):
            array = (
                np.random.uniform(0, 1, size=(20, 4096)) > .9).astype(np.double)
            fprints = [fp_type.from_vector(array[i, :])
                       for i in range(2)]
            folded_fprints = []
            for i, fp in enumerate(fprints):
                fp.name = str(i)
                folded_fprints.append(fp.fold(fold_len))
            db_fold1 = FingerprintDatabase(fp_type=fp_type)
            db_fold1.add_fingerprints(folded_fprints)

            db_unfold = FingerprintDatabase(fp_type=fp_type)
            db_unfold.add_fingerprints(fprints)

            db_fold2 = db_unfold.fold(fold_len)
            np.testing.assert_array_equal(db_fold2.array.todense().getA(),
                                          db_fold1.array.todense().getA())

    def test_update_props(self):
        from e3fp.fingerprint.fprint import CountFingerprint
        from e3fp.fingerprint.db import FingerprintDatabase
        array = (
            np.random.uniform(0, 1, size=(10, 1024)) > .9).astype(np.double)
        fprints = [CountFingerprint.from_vector(array[i, :])
                   for i in range(10)]
        for i, fp in enumerate(fprints):
            fp.name = str(i)
            fp.set_prop("index", i)
        db = FingerprintDatabase(fp_type=CountFingerprint)
        db.add_fingerprints(fprints)
        fprints2 = [CountFingerprint.from_vector(array[i, :])
                    for i in range(10)]
        for i, fp in enumerate(fprints2):
            fp.name = str(i + len(fprints))
            fp.set_prop("index", i)
        db.add_fingerprints(fprints2)
        indices = db.get_prop("index")
        self.assertEqual(indices.shape[0], 20)
        self.assertListEqual(indices.tolist(),
                             list(range(10)) + list(range(10)))

    def test_fingerprint_has_props(self):
        from e3fp.fingerprint.db import FingerprintDatabase
        array = (
            np.random.uniform(0, 1, size=(10, 1024)) > .9).astype(np.uint16)
        fp_names = [str(i) for i in range(10)]
        indices = [float(i) for i in range(10)]
        db = FingerprintDatabase.from_array(array, fp_names, level=5,
                                            name="Test",
                                            props={"index": indices})
        for i, x in enumerate(fp_names):
            self.assertEqual(db[x][0].get_prop("index"), indices[i])

    def test_append_dbs(self):
        from e3fp.fingerprint.db import append, FingerprintDatabase
        array = (
            np.random.uniform(0, 1, size=(10, 1024)) > .9).astype(np.uint16)
        fp_names = [str(i) for i in range(10)]
        indices = [float(i) for i in range(10)]
        dbs = []
        for i in range(10)[::2]:
            db = FingerprintDatabase.from_array(
                array[i:i + 2, :], fp_names[i:i + 2], level=5, name="Test",
                props={"index": indices[i:i + 2]})
            dbs.append(db)
        join_db = append(dbs)
        np.testing.assert_array_equal(join_db.array.todense().getA(), array)


if __name__ == "__main__":
    unittest.main()

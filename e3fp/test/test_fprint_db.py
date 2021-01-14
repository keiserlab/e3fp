"""Tests for fingerprint databases.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import pytest
import tempfile

import numpy as np


class TestFingerprintDatabase:

    """Tests for FingerprintDatabase"""

    def test_create(self):
        from e3fp.fingerprint.fprint import CountFingerprint
        from e3fp.fingerprint.db import FingerprintDatabase

        db = FingerprintDatabase(
            fp_type=CountFingerprint, level=5, name="TestDB"
        )
        assert db.fp_type is CountFingerprint
        assert db.level == 5
        assert db.name == "TestDB"

    def test_db_equality(self):
        from e3fp.fingerprint.fprint import Fingerprint, CountFingerprint
        from e3fp.fingerprint.db import FingerprintDatabase

        db1 = FingerprintDatabase(name="TestDB")
        db2 = FingerprintDatabase(name="TestDB2")
        assert db1 == db2
        db2 = FingerprintDatabase(fp_type=CountFingerprint, name="TestDB2")
        assert db1 != db2
        db2 = FingerprintDatabase(level=5, name="TestDB2")
        assert db1 != db2
        db2 = FingerprintDatabase(name="TestDB2")
        array = (np.random.uniform(0, 1, size=(10, 1024)) > 0.9).astype(
            np.uint16
        )
        fprints = [Fingerprint.from_vector(array[i, :]) for i in range(10)]
        for i, fp in enumerate(fprints):
            name = str(i)
            fp.name = name
        db1.add_fingerprints(fprints)
        db2.add_fingerprints(fprints)
        assert db1 == db2
        db2.add_fingerprints([fprints[0]])
        assert db1 != db2

    def test_create_from_array(self):
        from e3fp.fingerprint.fprint import Fingerprint, CountFingerprint
        from e3fp.fingerprint.db import FingerprintDatabase

        array = (np.random.uniform(0, 1, size=(10, 1024)) > 0.9).astype(
            np.uint16
        )
        fprints = [Fingerprint.from_vector(array[i, :]) for i in range(10)]
        fp_names = []
        for i, fp in enumerate(fprints):
            name = str(i)
            fp.name = name
            fp.level = 5
            fp_names.append(name)
        db1 = FingerprintDatabase(
            fp_type=CountFingerprint, level=5, name="Test"
        )
        db1.add_fingerprints(fprints)
        db2 = FingerprintDatabase.from_array(
            array, fp_names, level=5, name="Test"
        )
        assert db1.fp_type == db2.fp_type
        np.testing.assert_array_equal(
            db1.array.todense().getA(), db2.array.todense().getA()
        )

    def test_add_fingerprints(self):
        from e3fp.fingerprint.fprint import CountFingerprint
        from e3fp.fingerprint.db import FingerprintDatabase

        array = (np.random.uniform(0, 1, size=(10, 1024)) > 0.9).astype(
            np.double
        )
        fprints = [
            CountFingerprint.from_vector(array[i, :]) for i in range(10)
        ]
        for i, fp in enumerate(fprints):
            fp.name = str(i)
        db = FingerprintDatabase(fp_type=CountFingerprint)
        db.add_fingerprints(fprints)
        assert db.fp_type is CountFingerprint
        assert np.issubdtype(db.array.dtype, np.uint16)
        assert db.fp_num == 10
        assert db.bits == 1024
        assert len(db.fp_names_to_indices) == 10
        for k, v in db.fp_names_to_indices.items():
            k = int(k)
            assert k == v[0]
            np.testing.assert_almost_equal(
                array[k, :], db.array[k, :].todense().getA().flatten()
            )

    def test_add_fingerprints_two_batches(self):
        from e3fp.fingerprint.fprint import CountFingerprint
        from e3fp.fingerprint.db import FingerprintDatabase

        array = (np.random.uniform(0, 1, size=(10, 1024)) > 0.9).astype(
            np.double
        )
        fprints = [
            CountFingerprint.from_vector(array[i, :]) for i in range(10)
        ]
        for i, fp in enumerate(fprints):
            fp.name = str(i)
        db = FingerprintDatabase(fp_type=CountFingerprint)
        db.add_fingerprints(fprints)
        fprints2 = [
            CountFingerprint.from_vector(array[i, :]) for i in range(10)
        ]
        for i, fp in enumerate(fprints2):
            fp.name = str(i + len(fprints))
        db.add_fingerprints(fprints2)

        assert db.fp_type is CountFingerprint
        assert np.issubdtype(db.array.dtype, np.uint16)
        assert db.fp_num == 20
        assert db.bits == 1024
        assert len(db.fp_names_to_indices) == 20
        for k, v in db.fp_names_to_indices.items():
            k = int(k)
            assert k == v[0]
            np.testing.assert_almost_equal(
                array[k % 10, :], db.array[k, :].todense().getA().flatten()
            )

    def test_duplicate_fprint_names_detected(self):
        from e3fp.fingerprint.fprint import CountFingerprint
        from e3fp.fingerprint.db import FingerprintDatabase

        array = (np.random.uniform(0, 1, size=(10, 1024)) > 0.9).astype(
            np.double
        )
        fprints = [
            CountFingerprint.from_vector(array[i, :]) for i in range(10)
        ]
        for i, fp in enumerate(fprints):
            fp.name = str(i)
        db = FingerprintDatabase(fp_type=CountFingerprint)
        db.add_fingerprints(fprints)
        fprints2 = [
            CountFingerprint.from_vector(array[i, :]) for i in range(10)
        ]
        for i, fp in enumerate(fprints2):
            fp.name = str(i)
        db.add_fingerprints(fprints2)

        assert db.fp_type is CountFingerprint
        assert np.issubdtype(db.array.dtype, np.uint16)
        assert db.fp_num == 20
        assert db.bits == 1024
        assert len(db.fp_names_to_indices) == 10
        for k, v in db.fp_names_to_indices.items():
            k = int(k)
            assert k == v[0]
            assert k == v[1] - 10
            np.testing.assert_almost_equal(
                array[k % 10, :], db.array[k, :].todense().getA().flatten()
            )

    def test_get_db_subset(self):
        from e3fp.fingerprint.db import FingerprintDatabase

        array = (np.random.uniform(0, 1, size=(10, 1024)) > 0.9).astype(
            np.uint16
        )
        fp_names = []
        for i in range(array.shape[0]):
            fp_names.append(str(i))
        db = FingerprintDatabase.from_array(array, fp_names=fp_names, level=5)
        sub_db = db.get_subset(fp_names[:-2])
        assert db.level == sub_db.level
        assert db.fp_type == sub_db.fp_type
        assert (db.array[:-2, :] - sub_db.array).nnz == 0

    def test_roundtrip(self):
        """Ensure DB is the same after saving and loading."""
        from e3fp.fingerprint.db import FingerprintDatabase

        array = (np.random.uniform(0, 1, size=(10, 1024)) > 0.9).astype(
            np.uint16
        )
        fp_names = []
        for i in range(array.shape[0]):
            fp_names.append(str(i))
        db = FingerprintDatabase.from_array(
            array, fp_names=fp_names, level=5, props={"index": range(10)}
        )
        desc, db_file = tempfile.mkstemp(suffix=".fps.bz2")
        os.close(desc)
        db.save(db_file)
        db2 = db.load(db_file)
        os.unlink(db_file)
        assert db == db2
        assert db2.get_prop("index").tolist() == list(range(10))

    def test_roundtrip2(self):
        """Ensure DB is the same after saving and loading."""
        from e3fp.fingerprint.fprint import Fingerprint
        from e3fp.fingerprint.db import FingerprintDatabase

        fprints = []
        for i in range(10):
            fprints.append(
                Fingerprint.from_indices(
                    np.random.uniform(0, 2 ** 32, size=30),
                    bits=2 ** 32,
                    level=5,
                )
            )
            fprints[-1].name = "fp" + str(i)
            fprints[-1].set_prop("index", float(i))
        db = FingerprintDatabase(fp_type=Fingerprint, level=5)
        db.add_fingerprints(fprints)
        desc, db_file = tempfile.mkstemp(suffix=".fps.bz2")
        os.close(desc)
        db.save(db_file)
        db2 = db.load(db_file)
        os.unlink(db_file)
        assert db == db2
        assert db2.get_prop("index").tolist() == list(range(10))

    def test_roundtrip_zlib(self):
        """Ensure DB is the same after saving with savez and loading."""
        from e3fp.fingerprint.db import FingerprintDatabase

        array = (np.random.uniform(0, 1, size=(10, 1024)) > 0.9).astype(
            np.uint16
        )
        fp_names = []
        for i in range(array.shape[0]):
            fp_names.append(str(i))
        db = FingerprintDatabase.from_array(
            array, fp_names=fp_names, level=5, props={"index": range(10)}
        )
        desc, db_file = tempfile.mkstemp(suffix=".fpz")
        os.close(desc)
        db.savez(db_file)
        db2 = db.load(db_file)
        os.unlink(db_file)
        assert db == db2
        assert db2.get_prop("index").tolist() == list(range(10))

    def test_save_txt(self):
        """Ensure bitstrings saved to txt correctly."""
        from e3fp.fingerprint.db import FingerprintDatabase
        from python_utilities.io_tools import smart_open

        array = np.array(
            [[1, 0, 0, 1, 1], [0, 0, 0, 1, 0], [0, 1, 1, 1, 1]], dtype=np.bool_
        )
        db = FingerprintDatabase.from_array(array, ["1", "2", "3"])

        desc, txt_file = tempfile.mkstemp(suffix=".txt.gz")
        os.close(desc)
        db.savetxt(txt_file)
        exp_bitstring = b"10011 1\n00010 2\n01111 3\n"
        with smart_open(txt_file, "r") as f:
            bitstring = f.read()
        assert bitstring == exp_bitstring
        os.unlink(txt_file)

        desc, txt_file = tempfile.mkstemp(suffix=".txt.gz")
        os.close(desc)
        db.savetxt(txt_file, with_names=False)
        exp_bitstring = b"10011\n00010\n01111\n"
        with smart_open(txt_file, "r") as f:
            bitstring = f.read()
        assert bitstring == exp_bitstring
        os.unlink(txt_file)

    def test_save_txt_errors(self):
        """Check errors/warnings raised when saving bitstrings."""
        import warnings
        from scipy.sparse import csr_matrix
        from e3fp.util import E3FPEfficiencyWarning
        from e3fp.fingerprint.db import FingerprintDatabase
        from e3fp.fingerprint.fprint import CountFingerprint
        from e3fp.fingerprint.util import E3FPInvalidFingerprintError

        array = np.array(
            [[1, 0, 0, 1, 1], [0, 0, 0, 1, 0], [0, 1, 1, 1, 1]], dtype=np.bool_
        )
        db = FingerprintDatabase.from_array(
            array, ["1", "2", "3"], fp_type=CountFingerprint
        )

        desc, txt_file = tempfile.mkstemp(suffix=".txt.gz")
        os.close(desc)
        with pytest.raises(E3FPInvalidFingerprintError):
            db.savetxt(txt_file)

        array = csr_matrix((3, 2 ** 15), dtype=np.bool_)
        db = FingerprintDatabase.from_array(array, ["1", "2", "3"])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("error")

            with pytest.raises(E3FPEfficiencyWarning):
                db.savetxt(txt_file)

        os.unlink(txt_file)

    def test_load_efficiency_warning(self):
        import warnings
        from e3fp.util import E3FPEfficiencyWarning
        from e3fp.fingerprint.db import FingerprintDatabase
        import scipy

        array = (np.random.uniform(0, 1, size=(10, 1024)) > 0.9).astype(
            np.uint16
        )
        fp_names = []
        for i in range(array.shape[0]):
            fp_names.append(str(i))
        db = FingerprintDatabase.from_array(
            array, fp_names=fp_names, level=5, props={"index": range(10)}
        )
        desc, db_file = tempfile.mkstemp(suffix=".fpz")
        os.close(desc)
        db.savez(db_file)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("error")

            scipy.__version__ = "0.19"
            with pytest.raises(E3FPEfficiencyWarning):
                db.load(db_file)

            scipy.__version__ = "1.0"
            db.load(db_file)

        os.unlink(db_file)

    def test_lookup(self):
        from e3fp.fingerprint.fprint import Fingerprint
        from e3fp.fingerprint.db import FingerprintDatabase

        db = FingerprintDatabase(name="TestDB")
        array = (np.random.uniform(0, 1, size=(2, 1024)) > 0.9).astype(
            np.bool_
        )
        fp_names = []
        for i in range(array.shape[0]):
            fp_names.append("fp" + str(i))
        db = FingerprintDatabase.from_array(array, fp_names, name="Test")
        for i in range(array.shape[0]):
            assert Fingerprint.from_vector(array[i, :]) == db[i]
            assert Fingerprint.from_vector(array[i, :]) == db[db.fp_names[i]][0]

    def test_fold_db(self):
        from e3fp.fingerprint.fprint import Fingerprint, CountFingerprint
        from e3fp.fingerprint.db import FingerprintDatabase

        fold_len = 32
        for fp_type in (Fingerprint, CountFingerprint):
            array = (np.random.uniform(0, 1, size=(20, 4096)) > 0.9).astype(
                np.double
            )
            fprints = [fp_type.from_vector(array[i, :]) for i in range(2)]
            folded_fprints = []
            for i, fp in enumerate(fprints):
                fp.name = str(i)
                folded_fprints.append(fp.fold(fold_len))
            db_fold1 = FingerprintDatabase(fp_type=fp_type)
            db_fold1.add_fingerprints(folded_fprints)

            db_unfold = FingerprintDatabase(fp_type=fp_type)
            db_unfold.add_fingerprints(fprints)

            db_fold2 = db_unfold.fold(fold_len)
            np.testing.assert_array_equal(
                db_fold2.array.todense().getA(),
                db_fold1.array.todense().getA(),
            )

    def test_update_props(self):
        from e3fp.fingerprint.fprint import CountFingerprint
        from e3fp.fingerprint.db import FingerprintDatabase

        array = (np.random.uniform(0, 1, size=(10, 1024)) > 0.9).astype(
            np.double
        )
        fprints = [
            CountFingerprint.from_vector(array[i, :]) for i in range(10)
        ]
        for i, fp in enumerate(fprints):
            fp.name = str(i)
            fp.set_prop("index", i)
        db = FingerprintDatabase(fp_type=CountFingerprint)
        db.add_fingerprints(fprints)
        fprints2 = [
            CountFingerprint.from_vector(array[i, :]) for i in range(10)
        ]
        for i, fp in enumerate(fprints2):
            fp.name = str(i + len(fprints))
            fp.set_prop("index", i)
        db.add_fingerprints(fprints2)
        indices = db.get_prop("index")
        assert indices.shape[0] == 20
        assert indices.tolist() == (list(range(10)) + list(range(10)))

    def test_fingerprint_has_props(self):
        from e3fp.fingerprint.db import FingerprintDatabase

        array = (np.random.uniform(0, 1, size=(10, 1024)) > 0.9).astype(
            np.uint16
        )
        fp_names = [str(i) for i in range(10)]
        indices = [float(i) for i in range(10)]
        db = FingerprintDatabase.from_array(
            array, fp_names, level=5, name="Test", props={"index": indices}
        )
        for i, x in enumerate(fp_names):
            assert db[x][0].get_prop("index") == indices[i]

    def test_concat_dbs(self):
        from e3fp.fingerprint.db import concat, FingerprintDatabase

        array = (np.random.uniform(0, 1, size=(10, 1024)) > 0.9).astype(
            np.uint16
        )
        fp_names = [str(i) for i in range(10)]
        indices = [float(i) for i in range(10)]
        dbs = []
        for i in range(10)[::2]:
            db = FingerprintDatabase.from_array(
                array[i : i + 2, :],
                fp_names[i : i + 2],
                level=5,
                name="Test",
                props={"index": indices[i : i + 2]},
            )
            dbs.append(db)
        join_db = concat(dbs)
        np.testing.assert_array_equal(join_db.array.todense().getA(), array)

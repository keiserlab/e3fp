"""Tests for E3FP fingerprints.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import unittest


class FingerprintIOTestCases(unittest.TestCase):
    def test_fprint_from_indices(self):
        from e3fp.fingerprint.fprint import (
            Fingerprint,
            CountFingerprint,
            FloatFingerprint,
        )

        for fp_type in (Fingerprint, CountFingerprint, FloatFingerprint):
            in_indices = [3, 1, 4, 5]
            bits = 32
            fprint = fp_type.from_indices(in_indices, bits=bits)
            self.assertEqual(sorted(in_indices), sorted(fprint.indices))

    def test_fprint_from_fprint(self):
        from e3fp.fingerprint.fprint import (
            Fingerprint,
            CountFingerprint,
            FloatFingerprint,
        )

        for fp_type in (Fingerprint, CountFingerprint, FloatFingerprint):
            in_indices = [3, 1, 4, 5, 1, 5, 9]
            bits = 32
            fprint1 = fp_type.from_indices(in_indices, bits=bits)
            fprint2 = fp_type.from_fingerprint(fprint1)
            self.assertEqual(fprint1, fprint2)

    def test_countfprint_from_counts(self):
        from e3fp.fingerprint.fprint import CountFingerprint

        in_counts = {3: 1, 1: 4, 5: 1}
        bits = 32
        fprint = CountFingerprint.from_counts(in_counts, bits=bits)
        out_counts = fprint.counts
        self.assertEqual(in_counts, out_counts)

    def test_floatfprint_from_counts(self):
        from e3fp.fingerprint.fprint import FloatFingerprint

        in_counts = {3: 1.0, 1: 4.0, 5: 1.0}
        bits = 32
        fprint = FloatFingerprint.from_counts(in_counts, bits=bits)
        out_counts = fprint.counts
        self.assertEqual(in_counts, out_counts)

    def test_unique_indices(self):
        from e3fp.fingerprint.fprint import (
            Fingerprint,
            CountFingerprint,
            FloatFingerprint,
        )

        for fp_type in (Fingerprint, CountFingerprint, FloatFingerprint):
            in_indices = [3, 1, 4, 5, 1, 5, 9]
            bits = 32
            fprint = fp_type.from_indices(in_indices, bits=bits)
            self.assertEqual(sorted(set(in_indices)), sorted(fprint.indices))

    def test_bitstring_io(self):
        from e3fp.fingerprint.fprint import (
            Fingerprint,
            CountFingerprint,
            FloatFingerprint,
        )

        for fp_type in (Fingerprint, CountFingerprint, FloatFingerprint):
            in_bitstring = "1001001111011000"
            fprint = fp_type.from_bitstring(in_bitstring)
            out_bitstring = fprint.to_bitstring()
            self.assertEqual(in_bitstring, out_bitstring)

    def test_vector_io(self):
        from e3fp.fingerprint.fprint import (
            Fingerprint,
            CountFingerprint,
            FloatFingerprint,
        )
        import numpy as np

        for fp_type in (Fingerprint, CountFingerprint, FloatFingerprint):
            in_vector = np.array([0, 0, 1, 0, 1, 0, 1, 0, 0], dtype=np.bool_)
            fprint = fp_type.from_vector(in_vector)
            out_vector = fprint.to_vector(sparse=False)
            np.testing.assert_array_equal(in_vector, out_vector)

    def test_rdkit_io(self):
        from e3fp.fingerprint.fprint import (
            Fingerprint,
            CountFingerprint,
            FloatFingerprint,
        )

        for fp_type in (Fingerprint, CountFingerprint, FloatFingerprint):
            indices = [3, 1, 4, 5]
            bits = 32
            fprint1 = fp_type.from_indices(indices, bits=bits)
            rdkit_fprint1 = fprint1.to_rdkit()
            fprint2 = fp_type.from_rdkit(rdkit_fprint1)
            rdkit_fprint2 = fprint2.to_rdkit()
            self.assertEqual(rdkit_fprint1, rdkit_fprint2)

    def test_basic_properties(self):
        from e3fp.fingerprint.fprint import (
            Fingerprint,
            CountFingerprint,
            FloatFingerprint,
        )
        import numpy as np

        bits = 1024
        for i in range(10):
            indices = np.random.randint(0, bits, 30)
            unique_inds = np.unique(indices)
            level = int(np.random.randint(0, 10))
            for fp_type in (Fingerprint, CountFingerprint, FloatFingerprint):
                fp = fp_type.from_indices(indices, bits=bits, level=level)
                self.assertEqual(fp.bits, bits)
                self.assertEqual(len(fp), bits)
                self.assertEqual(fp.bit_count, unique_inds.size)
                self.assertAlmostEqual(
                    fp.density, float(unique_inds.size) / bits
                )


class FingerprintAlgebraTestCases(unittest.TestCase):
    pass


class FingerprintComparisonTestCases(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()

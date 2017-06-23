"""Integration tests for dependencies.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import unittest


class RequiredDependenciesTestCase(unittest.TestCase):
    def test_rdkit(self):
        import rdkit

    def test_numpy(self):
        import numpy

    def test_scipy(self):
        import scipy

    def test_murmurhash(self):
        import mmh3

    def test_python_utilities(self):
        import python_utilities


class OptionalFeatureDependenciesTestCase(unittest.TestCase):
    def test_h5py(self):
        import h5py

    def test_standardiser(self):
        import standardiser


class OptionalParallelDependenciesTestCase(unittest.TestCase):
    def test_mpi4py(self):
        import mpi4py

    def test_concurrent(self):
        import concurrent.futures

    def test_python_utilities(self):
        import python_utilities.parallel


if __name__ == "__main__":
    unittest.main()

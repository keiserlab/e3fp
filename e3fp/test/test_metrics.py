"""Tests for fingerprint comparison metrics.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import pytest

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from e3fp.fingerprint import metrics, fprint, db
from e3fp.fingerprint.metrics import array_metrics, fprint_metrics


def _create_random_sparse(nrows, nbits=1024, perc_pos=0.1, counts=False):
    arr = csr_matrix(
        np.random.uniform(0, 1, (nrows, nbits)) > (1 - perc_pos),
        dtype=np.double,
    )
    if counts:
        arr.data = np.random.randint(1, 30, arr.data.shape[0]).astype(
            np.double
        )
    return arr

def soergeldist(x, y):
    return np.abs(x - y).sum() / np.maximum(x, y).sum()


class TestArrayMetrics:

    """Tests for array comparison metrics"""

    @staticmethod
    def _eval(func, X, Y=None, dense=False, **kwargs):
        if dense:
            X = X.toarray()
            if Y is not None:
                Y = Y.toarray()
        return func(X, Y, **kwargs)

    @pytest.mark.parametrize("dense", [True, False])
    @pytest.mark.parametrize(
        "func,cdist_metric,counts",
        [
            (array_metrics.tanimoto, "jaccard", False),
            (array_metrics.dice, "dice", False),
            (array_metrics.cosine, "cosine", False),
            (array_metrics.cosine, "cosine", True),
            (array_metrics.pearson, "correlation", False),
            (array_metrics.pearson, "correlation", True),
            (array_metrics.soergel, soergeldist, False),
            (array_metrics.soergel, soergeldist, True),
        ],
    )
    def test_metrics_vs_cdist(self, func, cdist_metric, counts, dense):
        X = _create_random_sparse(10, counts=counts)
        Y = _create_random_sparse(8, counts=counts)
        expect_score = 1.0 - cdist(X.toarray(), Y.toarray(), metric=cdist_metric)
        score = self._eval(func, X, Y, dense=dense)
        assert type(score) is np.ndarray
        np.testing.assert_allclose(score, expect_score)
        # test self-comparison
        expect_score = 1.0 - cdist(X.toarray(), X.toarray(), metric=cdist_metric)
        score = self._eval(func, X, dense=dense)
        np.testing.assert_allclose(score, expect_score)

    @pytest.mark.parametrize("dense", [True, False])
    def test_tanimoto_soergel_equal_for_binary(self, dense):
        X = _create_random_sparse(10, counts=False)
        Y = _create_random_sparse(8, counts=False)
        tscore = self._eval(array_metrics.tanimoto, X, Y, dense=dense)
        sscore = self._eval(array_metrics.soergel, X, Y, dense=dense)
        np.testing.assert_allclose(tscore, sscore)


class TestFlexibleMetrics:

    """Tests for flexible comparison metrics"""

    metric_names = ["tanimoto", "soergel", "dice", "cosine", "pearson"]
    count_metric_names = ["soergel", "cosine", "pearson"]

    def test_binary_fprint_vs_fprint(self):
        fp1 = fprint.Fingerprint.from_vector(
            _create_random_sparse(1, counts=False, perc_pos=0.5)
        )
        fp2 = fprint.Fingerprint.from_vector(
            _create_random_sparse(1, counts=False, perc_pos=0.5)
        )
        for metric_name in self.metric_names:
            gen_score = getattr(metrics, metric_name)(fp1, fp2)
            fp_score = getattr(fprint_metrics, metric_name)(fp1, fp2)
            assert gen_score == pytest.approx(fp_score)
            array_score = getattr(array_metrics, metric_name)(
                fp1.to_vector(sparse=True), fp2.to_vector(sparse=True)
            )
            assert gen_score == pytest.approx(array_score[0][0])

    def test_count_fprint_vs_fprint(self):
        fp1 = fprint.CountFingerprint.from_vector(
            _create_random_sparse(1, nbits=32, counts=True, perc_pos=0.5)
        )
        fp2 = fprint.CountFingerprint.from_vector(
            _create_random_sparse(1, nbits=32, counts=True, perc_pos=0.5)
        )
        for metric_name in self.count_metric_names:
            gen_score = getattr(metrics, metric_name)(fp1, fp2)
            fp_score = getattr(fprint_metrics, metric_name)(fp1, fp2)
            assert gen_score == pytest.approx(fp_score)
            array_score = getattr(array_metrics, metric_name)(
                fp1.to_vector(sparse=True), fp2.to_vector(sparse=True)
            )
            assert gen_score == pytest.approx(array_score[0][0])

    def test_binary_fprint_vs_db(self):
        fp_array = _create_random_sparse(1, counts=False, perc_pos=0.5)
        fp = fprint.Fingerprint.from_vector(fp_array)
        db_array = _create_random_sparse(10, counts=False, perc_pos=0.5)
        fp_names = [str(i) for i in range(db_array.shape[0])]
        fdb = db.FingerprintDatabase.from_array(
            db_array, fp_names, fp_type=fprint.Fingerprint
        )
        for metric_name in self.metric_names:
            gen_score = getattr(metrics, metric_name)(fp, fdb)
            array_score = getattr(array_metrics, metric_name)(
                fp_array, db_array
            )
            np.testing.assert_allclose(gen_score, array_score)
            gen_score = getattr(metrics, metric_name)(fdb, fp)
            np.testing.assert_allclose(gen_score.T, array_score)

    def test_count_fprint_vs_db(self):
        fp_array = _create_random_sparse(1, counts=True, perc_pos=0.5)
        fp = fprint.CountFingerprint.from_vector(fp_array)
        db_array = _create_random_sparse(10, counts=True, perc_pos=0.5)
        fp_names = [str(i) for i in range(db_array.shape[0])]
        fdb = db.FingerprintDatabase.from_array(
            db_array, fp_names, fp_type=fprint.CountFingerprint
        )
        for metric_name in self.count_metric_names:
            gen_score = getattr(metrics, metric_name)(fp, fdb)
            array_score = getattr(array_metrics, metric_name)(
                fp_array, db_array
            )
            np.testing.assert_allclose(gen_score, array_score)
            # Check if reverse order produces transpose
            gen_score = getattr(metrics, metric_name)(fdb, fp)
            np.testing.assert_allclose(gen_score.T, array_score)

    def test_binary_db_vs_db(self):
        db_array1 = _create_random_sparse(1, counts=False, perc_pos=0.5)
        fp_names = [str(i) for i in range(db_array1.shape[0])]
        db1 = db.FingerprintDatabase.from_array(
            db_array1, fp_names, fp_type=fprint.Fingerprint
        )
        db_array2 = _create_random_sparse(1, counts=False, perc_pos=0.5)
        fp_names = [str(i) for i in range(db_array2.shape[0])]
        db2 = db.FingerprintDatabase.from_array(
            db_array2, fp_names, fp_type=fprint.Fingerprint
        )
        for metric_name in self.metric_names:
            gen_score = getattr(metrics, metric_name)(db1, db2)
            array_score = getattr(array_metrics, metric_name)(
                db_array1, db_array2
            )
            np.testing.assert_allclose(gen_score, array_score)

    def test_count_db_vs_db(self):
        db_array1 = _create_random_sparse(1, counts=True, perc_pos=0.5)
        fp_names = [str(i) for i in range(db_array1.shape[0])]
        db1 = db.FingerprintDatabase.from_array(
            db_array1, fp_names, fp_type=fprint.CountFingerprint
        )
        db_array2 = _create_random_sparse(1, counts=True, perc_pos=0.5)
        fp_names = [str(i) for i in range(db_array2.shape[0])]
        db2 = db.FingerprintDatabase.from_array(
            db_array2, fp_names, fp_type=fprint.CountFingerprint
        )
        for metric_name in self.count_metric_names:
            gen_score = getattr(metrics, metric_name)(db1, db2)
            array_score = getattr(array_metrics, metric_name)(
                db_array1, db_array2
            )
            np.testing.assert_allclose(gen_score, array_score)

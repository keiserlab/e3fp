"""Filtering and file generation methods for cross-validation.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import sys
import csv
import logging

import numpy as np
from sklearn import cross_validation as cv
from sklearn.metrics import auc as sk_auc

from seacore.util.library import SetValue
from python_utilities.io_tools import touch_dir
from e3fp.sea_utils.util import molecules_to_lists_dicts, \
                                lists_dicts_to_molecules, \
                                targets_to_dict, dict_to_targets, \
                                targets_to_mol_lists_targets, \
                                mol_lists_targets_to_targets, \
                                filter_molecules_by_targets, \
                                filter_targets_by_molecules

csv.field_size_limit(sys.maxsize)

OUT_CSV_EXT_DEF = ".csv.gz"


def dicts_to_cv_files(i, out_dir, targets_basename, molecules_basename,
                      group_type, targets_dict, mol_lists_dict,
                      smiles_dict, fp_type, overwrite=False,
                      out_ext=OUT_CSV_EXT_DEF):
    """Generate molecules/targets test/training files for cross-validation."""
    cv_targets_file = _make_cv_filename(out_dir, targets_basename, group_type,
                                        i, out_ext=out_ext)
    cv_molecules_file = _make_cv_filename(out_dir, molecules_basename,
                                          group_type, i, out_ext=out_ext)

    make_targets = make_molecules = False

    if overwrite or not os.path.isfile(cv_targets_file):
        make_targets = True
    if overwrite or not os.path.isfile(cv_molecules_file):
        make_molecules = True
    if not make_targets and not make_molecules:
        return (cv_targets_file, cv_molecules_file)

    touch_dir(_make_cv_subdir(out_dir, i))

    if make_targets:
        dict_to_targets(cv_targets_file, targets_dict)
        logging.info("Saved CV targets to {}.".format(cv_targets_file))
    if make_molecules:
        lists_dicts_to_molecules(cv_molecules_file, smiles_dict,
                                 mol_lists_dict, fp_type)
        logging.info("Saved CV molecules to {}.".format(cv_molecules_file))

    del mol_lists_dict
    return (cv_targets_file, cv_molecules_file)


def files_to_cv_files(targets_file, molecules_file, k=10, n=50,
                      affinity=None, out_dir=os.getcwd(), overwrite=False,
                      out_ext=OUT_CSV_EXT_DEF, split_by='targets'):
    """Generate molecules/targets test/training files for cross-validation."""

    if split_by not in ('targets', 'molecules'):
        raise ValueError(
            "Valid options for `split_by` are 'targets' and 'molecules'")

    targets_basename = _make_csv_basename(targets_file)
    molecules_basename = _make_csv_basename(molecules_file)

    logging.info("Reading and filtering targets.")
    mol_names_targets_dict = mol_lists_targets_to_targets(
        targets_to_dict(targets_file, affinity=affinity))
    targets_dict = filter_targets_by_molnum(mol_names_targets_dict, n=n)
    del mol_names_targets_dict
    logging.info("Reading molecules.")
    smiles_dict, mol_lists_dict, fp_type = molecules_to_lists_dicts(
        molecules_file)

    if split_by == 'targets':
        logging.info("Splitting target molecules into test/training sets.")
        train_test_targets = targets_to_cv_targets(targets_dict, k=k)
        train_test_mol_lists = [
            (filter_molecules_by_targets(mol_lists_dict, train),
             filter_molecules_by_targets(mol_lists_dict, test))
            for train, test in train_test_targets]
        # add negative data back in.
        neg_mol_names = set(mol_lists_dict.keys()).difference(
            *[x.cids for x in targets_dict.itervalues()])
        logging.info("{} molecules have no targets at threshold.".format(
            len(neg_mol_names)))
        if len(neg_mol_names) > 0:
            for i, (neg_train_mols,
                    neg_test_mols) in enumerate(data_to_train_test(
                        list(neg_mol_names), k=k)):
                train_test_mol_lists[i][0].update(
                    {mol_name: mol_lists_dict[mol_name] for mol_name
                     in neg_train_mols})
                train_test_mol_lists[i][1].update(
                    {mol_name: mol_lists_dict[mol_name] for mol_name
                     in neg_test_mols})
        train_test_targets = [
            (targets_to_mol_lists_targets(train, train_test_mol_lists[i][0]),
             targets_to_mol_lists_targets(test, train_test_mol_lists[i][1]))
            for i, (train, test) in enumerate(train_test_targets)]
    else:
        logging.info("Splitting all molecules into test/training sets.")
        train_test_mol_lists = mol_lists_to_cv_mol_lists(mol_lists_dict, k=k)
        train_test_targets = [
            (targets_to_mol_lists_targets(
                filter_targets_by_molecules(targets_dict, train), train),
             targets_to_mol_lists_targets(
                filter_targets_by_molecules(targets_dict, test), test))
            for i, (train, test) in enumerate(train_test_mol_lists)]
    del targets_dict
    # del mol_lists_dict

    logging.info("Saving cross-validation files.")
    for i in xrange(k):
        (train_mol_lists_dict, test_mol_lists_dict) = train_test_mol_lists[i]
        (train_targets_dict, test_targets_dict) = train_test_targets[i]

        (train_targets_file,
         train_molecules_file) = dicts_to_cv_files(
            i, out_dir, targets_basename, molecules_basename, "train",
            train_targets_dict, train_mol_lists_dict, smiles_dict, fp_type,
            out_ext=out_ext, overwrite=overwrite)
        logging.info("Saved training set files ({:d}/{:d}).".format(i+1, k))
        del train_targets_dict, train_mol_lists_dict

        (test_targets_file,
         test_molecules_file) = dicts_to_cv_files(
            i, out_dir, targets_basename, molecules_basename, "test",
            test_targets_dict, test_mol_lists_dict, smiles_dict, fp_type,
            out_ext=out_ext, overwrite=overwrite)
        logging.info("Saved test set files ({:d}/{:d}).".format(i+1, k))
        del test_targets_dict, test_mol_lists_dict

        yield (train_targets_file, train_molecules_file,
               test_targets_file, test_molecules_file)


def data_to_train_test(data, k=10):
    """Split data into `k` train/test sets."""
    data = np.asanyarray(data)
    kf = cv.KFold(data.shape[0], n_folds=k, shuffle=True)
    return ((data[train], data[test]) for train, test in kf)


def mol_lists_to_cv_mol_lists(mol_lists_dict, k=10):
    """Build `k` train/test mol list dicts from a mol list dict."""
    train_mol_lists_dict = [{} for i in xrange(k)]
    test_mol_lists_dict = [{} for i in xrange(k)]
    len(mol_lists_dict)
    for i, (train_mols, test_mols) in enumerate(
            data_to_train_test(mol_lists_dict.keys(), k=k)):
        train_mol_lists_dict[i] = {mol_id: mol_lists_dict[mol_id]
                                   for mol_id in train_mols}
        test_mol_lists_dict[i] = {mol_id: mol_lists_dict[mol_id]
                                  for mol_id in test_mols}
    return zip(train_mol_lists_dict, test_mol_lists_dict)


def targets_to_cv_targets(targets_dict, k=10):
    """Build `k` train/test target dicts from a target_dict."""
    train_targets_dicts = [{} for i in xrange(k)]
    test_targets_dicts = [{} for i in xrange(k)]
    for target_key, set_value in targets_dict.iteritems():
        for i, (train_cids, test_cids) in enumerate(
                data_to_train_test(set_value.cids, k=k)):
            train_targets_dicts[i][target_key] = SetValue(
                set_value.name, train_cids.tolist(), set_value.description)
            test_targets_dicts[i][target_key] = SetValue(
                set_value.name, test_cids.tolist(), set_value.description)
    return zip(train_targets_dicts, test_targets_dicts)


def filter_targets_by_molnum(targets_dict, n):
    """Return targets that have at least `n` binders."""
    return dict([(k, v) for k, v in targets_dict.iteritems()
                 if len(v.cids) >= n])


def merge_dicts(*ds):
    """Given dicts with (key, value), merge to dict with (key, [values])."""
    merged = {}
    for d in ds:
        for k, v in d.iteritems():
            merged.setdefault(k, []).append(v)
    return merged


def average_dict_values(*ds):
    """Given dicts with (key, value), return dict with (key, mean_value)."""
    merged = merge_dicts(*ds)
    return {k: np.mean(v) for k, v in merged.iteritems()}


def auc_dict_from_fp_tp_dict(d):
    """Calculate AUC from dict with tuple of FP rate and TP rate"""
    return {k: get_auc(v[0], v[1]) for k, v in d.iteritems()}


def logauc_dict_from_fp_tp_dict(d):
    """Calculate logAUC from dict with tuple of FP rate and TP rate"""
    return {k: get_logauc(v[0], v[1]) for k, v in d.iteritems()}


def get_delta_auc_dict(d1, d2):
    """Given 2 AUC dicts, subtract the AUC of the second from the first."""
    return {k: (v - d2[k]) for k, v in d1.iteritems() if k in d2}


def get_auc(fp, tp, adjusted=False):
    """Calculate AUC from the FP and TP arrays of an ROC curve."""
    # y = np.asarray(tp, dtype=np.double)
    # x = np.asarray(fp, dtype=np.double)
    # areas = (y[1:] + y[:-1]) * (x[1:] - x[:-1]) / 2.
    # auc = np.sum(areas)
    auc_val = sk_auc(fp, tp)
    if adjusted:
        auc_val -= 0.5
    return auc_val


def get_logauc(fp, tp, min_fp=0.001, adjusted=False):
    """Calculate logAUC, the AUC of the semilog ROC curve.

    `logAUC_lambda` is defined as the AUC of the ROC curve where the x-axis
    is in log space. In effect, this zooms the ROC curve onto the earlier
    portion of the curve where various classifiers will usually be
    differentiated. The adjusted logAUC is the logAUC minus the logAUC of
    a random classifier, resulting in positive values for better-than-random
    and negative otherwise.

    Reference:
        - Mysinger et al. J. Chem. Inf. Model. 2010, 50, 1561-1573.
    """
    lam_index = np.searchsorted(fp, min_fp)
    y = np.asarray(tp[lam_index:], dtype=np.double)
    x = np.asarray(fp[lam_index:], dtype=np.double)
    if (lam_index != 0):
        y = np.insert(y, 0, tp[lam_index - 1])
        x = np.insert(x, 0, min_fp)

    dy = (y[1:] - y[:-1])
    with np.errstate(divide='ignore'):
        intercept = y[1:] - x[1:] * (dy / (x[1:] - x[:-1]))
        intercept[np.isinf(intercept)] = 0.
    norm = np.log10(1. / float(min_fp))
    areas = ((dy / np.log(10.)) + intercept * np.log10(x[1:] / x[:-1])) / norm
    logauc = np.sum(areas)
    if adjusted:
        logauc -= 0.144620062  # random curve logAUC
    return logauc


def get_youden(fp, tp):
    """Get Youden's index (height of ROC above random) for each point."""
    return tp - fp


def get_youden_index(fp, tp, return_coordinate=False):
    """Calculate Youden's J statistic from ROC curve.

    Youden's J statistic is defined as the maximum high of an ROC curve above
    the diagonal. Symbolically,
        J = max{TPR(FPR) - FPR}
    """
    youden = get_youden(fp, tp)
    index = np.argmax(youden)
    if return_coordinate:
        return youden[index], (fp[index], tp[index])
    else:
        return youden[index]


def _make_cv_subdir(basedir, i):
    """Return cross-validation subdirectory."""
    return os.path.join(basedir, str(i))


def _make_cv_filename(out_dir, basename, group_type, i,
                      out_ext=OUT_CSV_EXT_DEF):
    """Return cross-validation filename for CSV file."""
    return os.path.join(_make_cv_subdir(out_dir, i),
                        "{!s}_{!s}_{:d}{!s}".format(basename, group_type, i,
                                                    out_ext))


def _make_csv_basename(filename):
    """Return basename of CSV file."""
    basename = os.path.splitext(os.path.basename(filename))[0]
    if basename.endswith(".csv"):
        basename = os.path.splitext(basename)[0]
    return basename

"""Filtering and file generation methods for cross-validation.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import sys
import csv
from itertools import izip
import logging

import numpy as np
from sklearn import cross_validation as cv

from seacore.util.library import SetValue
from python_utilities.io_tools import touch_dir
from e3fp.sea_utils.util import molecules_to_lists_dicts, \
                                lists_dicts_to_molecules, \
                                targets_to_dict, dict_to_targets, \
                                targets_to_mol_lists_targets, \
                                mol_lists_targets_to_targets, \
                                filter_molecules_by_targets

csv.field_size_limit(sys.maxsize)

OUT_CSV_EXT_DEF = ".csv.gz"


def dicts_to_cv_files(i, out_dir, targets_basename, molecules_basename,
                      group_type, group_targets_dict, mol_lists_dict,
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

    group_mol_lists_dict = filter_molecules_by_targets(mol_lists_dict,
                                                       group_targets_dict)
    if make_targets:
        group_mol_lists_targets_dict = targets_to_mol_lists_targets(
            group_targets_dict, group_mol_lists_dict)
        dict_to_targets(cv_targets_file, group_mol_lists_targets_dict)
        logging.info("Saved CV targets to %s." % cv_targets_file)
        del group_mol_lists_targets_dict
    if make_molecules:
        lists_dicts_to_molecules(cv_molecules_file, smiles_dict,
                                 group_mol_lists_dict, fp_type)
        logging.info("Saved CV molecules to %s." % cv_molecules_file)

    del group_mol_lists_dict
    return (cv_targets_file, cv_molecules_file)


def files_to_cv_files(targets_file, molecules_file, k=10, n=50,
                      affinity=None, out_dir=os.getcwd(), overwrite=False,
                      out_ext=OUT_CSV_EXT_DEF):
    """Generate molecules/targets test/training files for cross-validation."""
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

    logging.info("Making cross-validation target groups.")
    cv_targets_iter = targets_to_cv_targets(targets_dict, k=k)
    del targets_dict

    logging.info("Saving cross-validation files.")
    for i, (train_targets_dict,
            test_targets_dict) in enumerate(cv_targets_iter):

        (train_targets_file,
         train_molecules_file) = dicts_to_cv_files(
            i, out_dir, targets_basename, molecules_basename, "train",
            train_targets_dict, mol_lists_dict, smiles_dict, fp_type,
            out_ext=out_ext, overwrite=overwrite)
        logging.info("Saved training set files (%d/%d)." % (i+1, k))
        del train_targets_dict

        (test_targets_file,
         test_molecules_file) = dicts_to_cv_files(
            i, out_dir, targets_basename, molecules_basename, "test",
            test_targets_dict, mol_lists_dict, smiles_dict, fp_type,
            out_ext=out_ext, overwrite=overwrite)
        logging.info("Saved test set files (%d/%d)." % (i+1, k))
        del test_targets_dict

        yield (train_targets_file, train_molecules_file,
               test_targets_file, test_molecules_file)


def data_to_train_test(data, k=10):
    """Split data into `k` train/test sets."""
    data = np.asanyarray(data)
    kf = cv.KFold(data.shape[0], n_folds=k, shuffle=True)
    return ((data[train], data[test]) for train, test in kf)


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
    return izip(train_targets_dicts, test_targets_dicts)


def filter_targets_by_molnum(targets_dict, n):
    """Return targets that have at least `n` binders."""
    return dict([(k, v) for k, v in targets_dict.iteritems()
                 if len(v.cids) >= n])


def _make_cv_subdir(basedir, i):
    """Return cross-validation subdirectory."""
    return os.path.join(basedir, str(i))


def _make_cv_filename(out_dir, basename, group_type, i,
                      out_ext=OUT_CSV_EXT_DEF):
    """Return cross-validation filename for CSV file."""
    return os.path.join(
        _make_cv_subdir(out_dir, i), "%s_%s_%d%s" % (basename, group_type, i,
                                                     out_ext))


def _make_csv_basename(filename):
    """Return basename of CSV file."""
    basename = os.path.splitext(os.path.basename(filename))[0]
    if basename.endswith(".csv"):
        basename = os.path.splitext(basename)[0]
    return basename

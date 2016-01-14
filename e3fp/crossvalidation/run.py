"""Run cross-validation using SEA.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import logging
import cPickle as pickle

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from python_utilities.scripting import setup_logging
from python_utilities.io_tools import touch_dir, smart_open
from e3fp.sea_utils.library import build_library
from e3fp.sea_utils.util import molecules_to_lists_dicts, \
                                mol_lists_targets_to_targets, \
                                targets_to_dict
from e3fp.sea_utils.run import sea_set_search
from e3fp.crossvalidation.util import files_to_cv_files


setup_logging(reset=False)


def files_to_auc(targets_file, molecules_file, k=10, min_mols=50,
                 affinity=10000, cv_type="targets", targets_name="targets",
                 library_name="library", out_dir=os.getcwd(), overwrite=False,
                 auc_file="aucs.pkl.gz", roc_file="rocs.pkl.gz",
                 parallelizer=None):
    """Run k-fold cross-validation on input files.

    Parameters
    ----------
    targets_file : str
        All targets
    molecules_file : str
        All molecules
    k : int, optional
        Number of test/training data pairs to cross-validate.
    min_mols : int, optional
        Minimum number of molecules binding to a target for it to be
        considered.
    affinity : int, optional
        Affinity level of binders to consider.
    cv_type : str, optional
        Type of cross-validation to run. Options are 'targets' or 'molecules'.
    targets_name : str, optional
        Targets filename prefix.
    library_name : str, optional
        Library filename prefix
    out_dir : Attribute, optional
        Directory to which to write output files.
    overwrite : bool, optional
        Overwrite files.
    auc_file : str, optional
        Filename to which to write AUCs dict.
    roc_file : str, optional
        Filename to which to write ROCs dict.
    parallelizer : Parallelizer or None, optional
        Parallelizer for running the k CVs in parallel.

    Returns
    -------
    mean_auc : float
        Average AUC over all k CV runs.
    """
    touch_dir(out_dir)

    library_file = os.path.join(out_dir, "{!s}.sea".format(library_name))
    fit_file = os.path.join(out_dir, "{!s}.fit".format(library_name))
    if overwrite or not (os.path.isfile(library_file) and
                         os.path.isfile(fit_file)):
        logging.info("Generating background fit.")
        generate_fit = True
        if os.path.isfile(fit_file):
            logging.warning("Fit file already exists. Will not generate fit.")
            generate_fit = False
        build_library(library_file, molecules_file, targets_file, fit_file,
                      generate_fit=generate_fit)

    logging.info("Writing cross-validation files.")
    cv_files_iter = files_to_cv_files(targets_file, molecules_file, k=k,
                                      n=min_mols, affinity=affinity,
                                      out_dir=out_dir, overwrite=overwrite)

    args_list = []
    for i, (train_targets_file,
            train_molecules_file,
            test_targets_file,
            test_molecules_file) in enumerate(cv_files_iter):
        msg = " ({:d} / {:d})".format(i + 1, k)
        cv_dir = os.path.dirname(train_targets_file)
        train_library_file = os.path.join(cv_dir, "train_{!s}.sea".format(
            library_name))
        cv_auc_file = os.path.join(cv_dir, auc_file)
        cv_roc_file = os.path.join(cv_dir, roc_file)
        args_list.append((molecules_file, test_targets_file,
                          test_molecules_file, train_targets_file,
                          train_molecules_file, train_library_file,
                          cv_auc_file, cv_roc_file, fit_file, cv_type, msg,
                          overwrite))

    if parallelizer is not None:
        mean_aucs = np.asarray(
            zip(*parallelizer.run(run_cv, iter(args_list)))[0],
            dtype=np.float)
    else:
        mean_aucs = np.asarray([run_cv(*x) for x in args_list],
                               dtype=np.float)

    return np.mean(mean_aucs)


def run_cv(molecules_file, test_targets_file, test_molecules_file,
           train_targets_file, train_molecules_file, train_library_file,
           auc_file, roc_file, fit_file, cv_type="targets", msg="",
           overwrite=False):
    """Run a single cross-validation on input training/test files.

    Parameters
    ----------
    molecules_file : str
        All molecules
    test_targets_file : str
        Targets and corresponding molecules in test set
    test_molecules_file : str
        Test molecules (usually not disjoint from training molecules)
    train_targets_file : str
        Targets and corresponding molecules in training set
    train_molecules_file : str
        Training molecules (usually not disjoint from test molecules)
    train_library_file : str
        SEA library for training data.
    auc_file : str
        File to which to write AUCs.
    roc_file : str
        File to which to write ROCs.
    fit_file : str
        SEA fit file.
    msg : str, optional
        Message to append to log messages.
    overwrite : bool, optional
        Overwrite files.
    cv_type : str, optional
        Type of cross-validation to run. Options are 'targets' or 'molecules'.

    Returns
    -------
    mean_auc : float
        Average AUC over targets or molecules (depending on `type`).
    """
    if overwrite or not os.path.isfile(train_library_file):
        logging.info("Building library for training set.{}".format(msg))
        build_library(train_library_file, train_molecules_file,
                      train_targets_file, fit_file, generate_fit=False)

    if (os.path.isfile(auc_file) and os.path.isfile(roc_file) and
            not overwrite):
        logging.info("Loading CV results from files.{}".format(msg))
        with smart_open(auc_file, "rb") as f:
            aucs_dict = pickle.load(f)
        with smart_open(roc_file, "rb") as f:
            fp_tp_rates_dict = pickle.load(f)
    else:
        logging.info(
            "Searching test sequences against library.{}".format(msg))
        fp_tp_rates_dict, aucs_dict = cv_files_to_roc_auc(
            molecules_file, test_targets_file, test_molecules_file,
            train_targets_file, train_library_file, cv_type=cv_type)

        if overwrite or not os.path.isfile(auc_file):
            with smart_open(auc_file, "wb") as f:
                pickle.dump(aucs_dict, f)

        if overwrite or not os.path.isfile(roc_file):
            with smart_open(roc_file, "wb") as f:
                pickle.dump(fp_tp_rates_dict, f)

    mean_auc = np.mean(aucs_dict.values())
    logging.info("Mean AUC: {:.4f}{}.".format(mean_auc, msg))
    return mean_auc


def cv_files_to_roc_auc(molecules_file, test_targets_file,
                        test_molecules_file, train_targets_file,
                        train_library_file, affinity=10000,
                        cv_type="targets"):
    """Get ROCs dict (false/true positives) and AUCs dict.

    Parameters
    ----------
    molecules_file : str
        All molecules
    test_targets_file : str
        Targets and corresponding molecules in test set
    test_molecules_file : str
        Test molecules (usually not disjoint from training molecules)
    train_targets_file : str
        Targets and corresponding molecules in training set
    train_library_file : str
        SEA library for training data.
    affinity : int, optional
        Affinity level of binders to consider.
    cv_type : str, optional
        Type of cross-validation to run. Options are 'targets' or 'molecules'.

    Returns
    -------
    fp_tp_rates_dict : dict
        Dict matching a key to an 2xN array with false positive rates and true
        positive rates at the same N thresholds. If `cv_type` is 'targets',
        key is a ``TargetKey``. If `cv_type` is 'molecules', key is a molecule
        id. The rows of the array, when plotted against each other, form an
        ROC curve.
    aucs_dict : dict
        Dict matching key (same as in `fp_tp_rates_dict`) to AUC for ROC
        curve.
    """
    aucs_dict = {}
    fp_tp_rates_dict = {}
    _, mol_lists_dict, _ = molecules_to_lists_dicts(molecules_file)
    all_molecules = set(mol_lists_dict.keys())
    del mol_lists_dict

    _, test_mol_lists_dict, _ = molecules_to_lists_dicts(test_molecules_file)
    del _

    if cv_type == "molecules":
        raise NotImplemented(
            "Cross-validation with targets doesn't exist yet.")
    elif cv_type != "targets":
        raise ValueError(
            "Valid options for `cv_type` are 'targets' and 'molecules'")

    logging.info("Searching {:d} fingerprints against {}.".format(
        len(test_mol_lists_dict), train_library_file))
    results = sea_set_search(train_library_file, test_mol_lists_dict,
                             log=True)
    logging.info("Calculating ROC curves and AUCs.")
    train_targets_dict = mol_lists_targets_to_targets(
        targets_to_dict(train_targets_file))
    test_targets_dict = mol_lists_targets_to_targets(
        targets_to_dict(test_targets_file))
    for target_key in train_targets_dict.iterkeys():
        all_hit_mols = results.target_result(target_key)
        trained_mols = set(train_targets_dict[target_key].cids)
        tested_mols = all_molecules.difference(trained_mols)
        tested_hit_mols = all_hit_mols.difference(trained_mols)
        test_true_mols = set(test_targets_dict[target_key].cids)

        truepos_hit_mols = tested_hit_mols.intersection(test_true_mols)
        falsepos_hit_mols = tested_hit_mols.difference(test_true_mols)
        true_false = np.array([x in test_true_mols
                               for x in tested_mols], dtype=np.int)

        evalues, max_tc = (np.array(y, dtype=np.float)
                           for y in zip(
                              *[results.mol_result(x).get(target_key, (1, 0))
                                for x in tested_mols])
                           )

        inf_ind = np.where(evalues == 0)
        log10e = -np.log10(evalues)
        real_ind = np.where(evalues != 0)
        if real_ind[0].shape[0] == 0:
            real_max = 1.
        else:
            real_max = np.amax(log10e[real_ind])
        log10e[inf_ind] = real_max + 1.

        if len(truepos_hit_mols) == 0:
            if len(falsepos_hit_mols) > 0:
                fpr = np.array([0., 1.], dtype=np.float)
                tpr = np.array([0., 0.], dtype=np.float)
                roc_auc = 0.
            else:
                continue
        else:
            fpr, tpr, thresholds = roc_curve(true_false, log10e)

        roc_auc = auc(fpr, tpr)
        fp_tp_rates_dict[target_key] = np.array([fpr, tpr], dtype=np.float)
        aucs_dict[target_key] = roc_auc

    return fp_tp_rates_dict, aucs_dict

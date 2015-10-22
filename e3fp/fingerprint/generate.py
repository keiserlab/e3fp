"""Generate E3FP fingerprints.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
from __future__ import division, print_function
import os
import logging
import argparse

from python_utilities.scripting import setup_logging
from python_utilities.parallel import make_data_iterator, Parallelizer, \
                                        ALL_PARALLEL_MODES
from python_utilities.io_tools import touch_dir
from e3fp.conformer.util import mol_from_sdf, MolItemName
from e3fp.fingerprint.fprinter import Fingerprinter
import e3fp.fingerprint.fprint as fp


def fprints_dict_from_sdf(sdf_file, **kwargs):
    try:
        mol = mol_from_sdf(sdf_file)
    except:
        logging.error("Error retrieving mol from %s." % (sdf_file))
        return False
    fprints_dict = fprints_dict_from_mol(mol, **kwargs)
    return fprints_dict


def fprints_dict_from_mol(mol, level=-1, shell_radius=2.0, first=-1,
                          counts=False, stereo=False, out_dir_base="E3FP",
                          out_ext=".bz2", store_identifiers_map=False,
                          include_disconnected=True, overwrite=False,
                          save=True, all_iters=False):
    """Build a E3FP fingerprint from a mol encoded in an SDF file.

    Parameters
    ----------
    sdf_file : str
        SDF file path.
    level : int, optional (default -1)
        Level/maximum number of iterations of E3FP. If -1 is provided, it runs
        until termination, and `all_iters` is set to False.
    shell_radius : float, optional (default 2.0)
        Radius multiplier for spherical shells.
    first : int, optional (default -1)
        First `N` number of conformers from file to fingerprint. If -1, all are
        fingerprinted.
    counts : bool (default False)
        Instead of bit-based ``Fingerprint`` objects, generate count-based
        ``CountFingerprint`` objects.
    stereo : bool, optional (default False)
        Incorporate stereochemistry in fingerprint.
    out_dir_base : str, optional (default "E3FP")
        Basename of out directory to save fingerprints. Iteration number is
        appended.
    out_ext : str, optional (default ".bz2")
        Extension on fingerprint pickles, used to determine compression level.
    store_identifiers_map : bool, optional (default False)
        Within each fingerprint, store map from each identifier to
        corresponding substructure. Drastically increases size of fingerprint.
    include_disconnected : bool, optional (default True)
        Include disconnected atoms when hashing and for stereo calculations.
        Turn off purely for testing purposes, to make E3FP more like ECFP.
    overwrite : bool, optional (default False)
        Overwrite pre-existing file.
    save : bool, optional (default True)
        Save fingerprints to directory.
    all_iters : bool, optional (default True)
        Save fingerprints from all iterations to file(s).
    """
    name = mol.GetProp("_Name")

    if level is None:
        level = -1

    if save:
        filenames = []
        all_files_exist = True
        if level == -1 or not all_iters:
            if level == -1:
                dir_name = "%s_complete" % (out_dir_base)
            else:
                dir_name = "%s%d" % (out_dir_base, level)
            touch_dir(dir_name)
            filenames.append("%s/%s%s" % (dir_name, name, out_ext))
            if not os.path.isfile(filenames[0]):
                all_files_exist = False
        else:
            for i in xrange(level + 1):
                dir_name = "%s%d" % (out_dir_base, i)
                touch_dir(dir_name)
                filename = "%s/%s%s" % (dir_name, name, out_ext)
                filenames.append(filename)
                if not os.path.isfile(filename):
                    all_files_exist = False

        if all_files_exist and not overwrite:
            logging.warning("All fingerprint files for %s already exist. Skipping." % (name))
            return {}

    fingerprinter = Fingerprinter(level=level, radius_multiplier=shell_radius,
                                  counts=counts, stereo=stereo,
                                  store_identifiers_map=store_identifiers_map,
                                  include_disconnected=include_disconnected)

    try:
        fprints_dict = {}
        logging.info("Generating fingerprints for %s." % name)
        term_iter = level
        for j, conf in enumerate(mol.GetConformers()):
            if j == first:
                j -= 1
                break
            fingerprinter.run(conf=conf)
            term_iter = max(max(fingerprinter.identifiers_at_level.keys()),
                            term_iter, level)
            for i in xrange(term_iter + 1):
                fprint = fingerprinter.get_fingerprint_at_level(i)
                fprint.name = MolItemName.from_str(name).to_conf_name(j)
                if i not in fprints_dict and j != 0:
                    fprints_dict[i] = fprints_dict[i-1][:j]
                fprints_dict.setdefault(i, []).append(fprint)
        logging.info("Generated %d fingerprints for %s." % (j + 1, name))
    except:
        logging.error("Error generating fingerprints for %s." % (name),
                      exc_info=True)
        return {}

    if save:
        if level == -1 or not all_iters:
            term_iter = max(fprints_dict.keys())
            fprints = fprints_dict[term_iter]
            try:
                fp.savez(filenames[0], *fprints)
                logging.info("Saved fingerprints for %s." % name)
            except Exception:
                logging.error(
                    "Error saving fingerprints for %s to %s" % (
                        name, filenames[0]), exc_info=True)
                return {}
        else:
            try:
                for i, fprints in sorted(fprints_dict.items()):
                    fp.savez(filenames[i], *fprints)
                logging.info("Saved fingerprints for %s." % name)
            except Exception:
                logging.error(
                    "Error saving fingerprints for %s to %s" % (
                        name, filenames[i]), exc_info=True)
                return {}

    return fprints_dict


def run(sdf_files, out_dir_base="E3FP", out_ext=".bz2", first=-1,
        level=-1, shell_radius=2.0, counts=False, stereo=False,
        store_identifiers_map=False, exclude_disconnected=False,
        overwrite=False, log=None, num_proc=None, parallel_mode=None,
        verbose=False, all_iters=False):
    """Generate E3FP fingerprints from SDF files.

    Parameters
    ----------
    sdf_files : str
        Path to SDF file(s), each with one molecule and multiple conformers.
    out_dir_base : str, optional (default "E3FP")
        Basename for output directory to save fingerprints. Iteration number
        is appended to basename.
    out_ext : str, optional (default ".bz2")
        Extension for fingerprint pickles. Options are (".pkl", ".gz", ".bz2").
    first : int, optional (default -1)
        Maximum number of first conformers for which to generate fingerprints.
    level : int, optional (default -1)
        Level/maximum number of iterations for fingerprint generation.
    shell_radius : float, optional (default 2.0)
        Distance to increment shell radius at around each atom, starting at
        0.0.
    counts : bool, optional (default False)
        Store counts-based E3FC instead of default bit-based.
    stereo : bool, optional (default False)
        Differentiate by stereochemistry.
    store_identifiers_map : bool, optional (default False)
        Within each fingerprint, store map from "on" bits to each substructure
        represented.
    exclude_disconnected : bool, optional (default False)
        Exclude disconnected atoms when hashing, but do use them for stereo
        calculations. Included purely for debugging, to make E3FP more like
        ECFP.
    overwrite : bool, optional (default False)
        Overwrite existing file(s).
    log : str, optional (default None)
        Log filename.
    num_proc : int, optional (default None)
        Set number of processors to use.
    verbose : bool, optional (default False)
        Run with extra verbosity.
    all_iters : bool, optional (default False)
        Save fingerprints from all iterations to file(s).
    """
    para = Parallelizer(num_proc=num_proc, parallel_mode=parallel_mode)

    setup_logging(log, verbose)

    if para.rank == 0:
        logging.info("Initializing E3FP generation.")
        logging.info("Getting SDF files")

        if len(sdf_files) == 1 and os.path.isdir(sdf_files[0]):
            from glob import glob
            sdf_files = glob("%s/*" % sdf_files[0])

        data_iterator = make_data_iterator(sdf_files)

        logging.info("SDF File Number: %d" % len(sdf_files))
        logging.info("Out Directory Basename: %s" % out_dir_base)
        logging.info("Out Extension: %s" % out_ext)
        logging.info("Max First Conformers: %d" % first)
        logging.info("Level/Max Iterations: %d" % level)
        logging.info("Shell Radius Multiplier: %.4g" % shell_radius)
        logging.info("Stereo Mode: %s" % str(stereo))
        if para.is_mpi:
            logging.info("Parallel Mode: %s" % para.parallel_mode)
        elif para.is_concurrent:
            logging.info("Parallel Mode: multiprocessing")
        else:
            logging.info("Parallel Mode: off")
        logging.info("Starting")
    else:
        data_iterator = iter([])

    fp_kwargs = {"first": int(first),
                 "level": int(level),
                 "shell_radius": float(shell_radius),
                 "stereo": stereo,
                 "out_dir_base": out_dir_base,
                 "out_ext": out_ext,
                 "counts": counts,
                 "overwrite": overwrite,
                 "store_identifiers_map": store_identifiers_map,
                 "include_disconnected": not exclude_disconnected,
                 "all_iters": all_iters}

    run_kwargs = {
        "kwargs": fp_kwargs, "logging_str": "Generated fingerprints for %s",
        "logging_format": lambda x: os.path.basename(x[0]).split(os.extsep)[0]}

    para.run(fprints_dict_from_sdf, data_iterator, **run_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        """Generate E3FP fingerprints from SDF files.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('sdf_files', nargs='+', type=str,
                        help="""Path to SDF file(s), each with one molecule
                             and multiple conformers.""")
    parser.add_argument('-o', '--out_dir_base', type=str,
                        default="E3FP",
                        help="""Basename for output directory to save
                             fingerprints. Iteration number is appended to
                             basename.""")
    parser.add_argument('--out_ext', type=str, default=".bz2",
                        choices=[".pkl", ".gz", ".bz2"],
                        help="""Extension for fingerprint pickles.""")
    parser.add_argument('--all_iters', action='store_true',
                        help="""Save fingerprints from all iterations to
                             file(s).""")
    parser.add_argument('--first', type=int, default=-1,
                        help="""Set maximum number of first conformers to
                             generare fingerprints for.""")
    parser.add_argument('-m', '--level', '--max_iterations', type=int,
                        default=-1,
                        help="""Maximum number of iterations for fingerprint
                             generation. If -1, fingerprinting is run until
                             termination, and `all_iters` is set to False.""")
    parser.add_argument('-r', '--shell_radius', type=float, default=2.0,
                        help="""Distance to increment shell radius at around
                             each atom, starting at 0.0.""")
    parser.add_argument('--counts', action='store_true',
                        help="""Store counts-based E3FC instead of default
                             bit-based.""")
    parser.add_argument('--stereo', action="store_true",
                        help="""Differentiate by stereochemistry.""")
    parser.add_argument('--store_identifiers_map', action="store_true",
                        help="""Within each fingerprint, store map from "on"
                             bits to each substructure represented.""")
    parser.add_argument('--exclude_disconnected', action="store_true",
                        help="""Include disconnected atoms when hashing, but
                             do use them for stereo calculations. Turn off
                             purely for debugging, to make E3FP more like
                             ECFP.""")
    parser.add_argument('-O', '--overwrite', action="store_true",
                        help="""Overwrite existing file(s).""")
    parser.add_argument('-l', '--log', type=str, default=None,
                        help="Log filename.")
    parser.add_argument('-p', '--num_proc', type=int, default=None,
                        help="""Set number of processors to use.""")
    parser.add_argument('--parallel_mode', type=str, default=None,
                        choices=list(ALL_PARALLEL_MODES),
                        help="""Set parallelization mode to use.""")
    parser.add_argument('-v', '--verbose', action="store_true",
                        help="Run with extra verbosity.")
    params = parser.parse_args()

    kwargs = dict(params._get_kwargs())
    sdf_files = kwargs.pop('sdf_files')
    run(sdf_files, **kwargs)

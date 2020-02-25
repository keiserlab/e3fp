"""Generate E3FP fingerprints.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
from __future__ import division, print_function
import os
import logging
import argparse
import sys

from python_utilities.scripting import setup_logging
from python_utilities.parallel import (
    make_data_iterator,
    Parallelizer,
    ALL_PARALLEL_MODES,
)
from python_utilities.io_tools import touch_dir
from e3fp.config.params import read_params, get_default_value, get_value
from e3fp.conformer.util import mol_from_sdf, MolItemName
from e3fp.fingerprint.fprinter import Fingerprinter, BITS
from e3fp.fingerprint.db import FingerprintDatabase
import e3fp.fingerprint.fprint as fp

LEVEL_DEF = get_default_value("fingerprinting", "level", int)
RADIUS_MULTIPLIER_DEF = get_default_value(
    "fingerprinting", "radius_multiplier", float
)
FIRST_DEF = get_default_value("fingerprinting", "first", int)
COUNTS_DEF = get_default_value("fingerprinting", "counts", bool)
STEREO_DEF = get_default_value("fingerprinting", "stereo", bool)
INCLUDE_DISCONNECTED_DEF = get_default_value(
    "fingerprinting", "include_disconnected", bool
)
RDKIT_INVARIANTS_DEF = get_default_value(
    "fingerprinting", "rdkit_invariants", bool
)
EXCLUDE_FLOATING_DEF = get_default_value(
    "fingerprinting", "exclude_floating", bool
)
REMOVE_DUPLICATE_SUBSTRUCTS_DEF = get_default_value(
    "fingerprinting", "remove_duplicate_substructs", bool
)
OUT_EXT_DEF = ".fp.bz2"


def fprints_dict_from_sdf(sdf_file, **kwargs):
    """Build fingerprints dict for conformers encoded in an SDF file.

    See `fprints_dict_from_mol` for description of arguments.
    """
    try:
        mol = mol_from_sdf(sdf_file)
    except Exception:
        logging.error("Error retrieving mol from {!s}.".format(sdf_file))
        return False
    fprints_dict = fprints_dict_from_mol(mol, **kwargs)
    return fprints_dict


def fprints_dict_from_mol(
    mol,
    bits=BITS,
    level=LEVEL_DEF,
    radius_multiplier=RADIUS_MULTIPLIER_DEF,
    first=FIRST_DEF,
    counts=COUNTS_DEF,
    stereo=STEREO_DEF,
    include_disconnected=INCLUDE_DISCONNECTED_DEF,
    rdkit_invariants=RDKIT_INVARIANTS_DEF,
    exclude_floating=EXCLUDE_FLOATING_DEF,
    remove_duplicate_substructs=REMOVE_DUPLICATE_SUBSTRUCTS_DEF,
    out_dir_base=None,
    out_ext=OUT_EXT_DEF,
    save=False,
    all_iters=False,
    overwrite=False,
):
    """Build a E3FP fingerprint from a mol with at least one conformer.

    Parameters
    ----------
    mol : RDKit Mol
        Input molecule with one or more conformers to be fingerprinted.
    bits : int
        Set number of bits for final folded fingerprint.
    level : int, optional
        Level/maximum number of iterations of E3FP. If -1 is provided, it runs
        until termination, and `all_iters` is set to False.
    radius_multiplier : float, optional
        Radius multiplier for spherical shells.
    first : int, optional
        First `N` number of conformers from file to fingerprint. If -1, all
        are fingerprinted.
    counts : bool, optional
        Instead of bit-based fingerprints. Otherwise, generate count-based
        fingerprints.
    stereo : bool, optional
        Incorporate stereochemistry in fingerprint.
    remove_duplicate_substructs : bool, optional
        If a substructure arises that corresponds to an identifier already in
        the fingerprint, then the identifier for the duplicate substructure is
        not added to fingerprint.
    include_disconnected : bool, optional
        Include disconnected atoms when hashing and for stereo calculations.
        Turn off purely for testing purposes, to make E3FP more like ECFP.
    rdkit_invariants : bool, optional
        Use the atom invariants used by RDKit for its Morgan fingerprint.
    exclude_floating : bool, optional:
        Mask atoms with no bonds (usually floating ions) from the fingerprint.
        These are often placed arbitrarily and can confound the fingerprint.
    out_dir_base : str, optional
        Basename of out directory to save fingerprints. Iteration number is
        appended.
    out_ext : str, optional
        Extension on fingerprint pickles, used to determine compression level.
    save : bool, optional
        Save fingerprints to directory.
    all_iters : bool, optional
        Save fingerprints from all iterations to file(s).
    overwrite : bool, optional
        Overwrite pre-existing file.

    Deleted Parameters
    ------------------
    sdf_file : str
        SDF file path.
    """
    name = mol.GetProp("_Name")

    if level is None:
        level = -1

    if bits in (-1, None):
        bits = BITS

    if save:
        filenames = []
        all_files_exist = True
        if level == -1 or not all_iters:
            if level == -1:
                dir_name = "{!s}_complete".format(out_dir_base)
            else:
                dir_name = "{!s}{:d}".format(out_dir_base, level)
            touch_dir(dir_name)
            filenames.append(
                os.path.join(dir_name, "{!s}{!s}".format(name, out_ext))
            )
            if not os.path.isfile(filenames[0]):
                all_files_exist = False
        else:
            for i in range(level + 1):
                dir_name = "{:s}{:d}".format(out_dir_base, i)
                touch_dir(dir_name)
                filename = os.path.join(
                    dir_name, "{!s}{!s}".format(name, out_ext)
                )
                filenames.append(filename)
                if not os.path.isfile(filename):
                    all_files_exist = False

        if all_files_exist and not overwrite:
            logging.warning(
                "All fingerprint files for {!s} already exist. "
                "Skipping.".format(name)
            )
            return {}

    fingerprinter = Fingerprinter(
        bits=bits,
        level=level,
        radius_multiplier=radius_multiplier,
        counts=counts,
        stereo=stereo,
        include_disconnected=include_disconnected,
        rdkit_invariants=rdkit_invariants,
        exclude_floating=exclude_floating,
        remove_duplicate_substructs=remove_duplicate_substructs,
    )

    try:
        fprints_dict = {}
        logging.info("Generating fingerprints for {!s}.".format(name))
        for j, conf in enumerate(mol.GetConformers()):
            if j == first:
                j -= 1
                break
            fingerprinter.run(conf, mol)
            # fingerprinter.save_substructs_to_db(substruct_db) #PLACEHOLDER
            level_range = range(level + 1)
            if level == -1 or not all_iters:
                level_range = (level,)
            else:
                level_range = range(level + 1)
            for i in level_range:
                fprint = fingerprinter.get_fingerprint_at_level(i)
                fprint.name = MolItemName.from_str(name).to_conf_name(j)
                # if i not in fprints_dict and j != 0:
                #     fprints_dict[i] = fprints_dict[i-1][:j]
                fprints_dict.setdefault(i, []).append(fprint)
        logging.info(
            "Generated {:d} fingerprints for {!s}.".format(j + 1, name)
        )
    except Exception:
        logging.error(
            "Error generating fingerprints for {:s}.".format(name),
            exc_info=True,
        )
        return {}

    if save:
        if level == -1 or not all_iters:
            fprints = fprints_dict[max(fprints_dict.keys())]
            try:
                fp.savez(filenames[0], *fprints)
                logging.info("Saved fingerprints for {:s}.".format(name))
            except Exception:
                logging.error(
                    "Error saving fingerprints for {:s} to {:s}".format(
                        name, filenames[0]
                    ),
                    exc_info=True,
                )
                return {}
        else:
            try:
                for i, fprints in sorted(fprints_dict.items()):
                    fp.savez(filenames[i], *fprints)
                logging.info("Saved fingerprints for {:s}.".format(name))
            except Exception:
                logging.error(
                    "Error saving fingerprints for {:s} to {:s}".format(
                        name, filenames[i]
                    ),
                    exc_info=True,
                )
                return {}

    return fprints_dict


def run(
    sdf_files,
    bits=BITS,
    first=FIRST_DEF,
    level=LEVEL_DEF,
    radius_multiplier=RADIUS_MULTIPLIER_DEF,
    counts=COUNTS_DEF,
    stereo=STEREO_DEF,
    include_disconnected=INCLUDE_DISCONNECTED_DEF,
    rdkit_invariants=RDKIT_INVARIANTS_DEF,
    exclude_floating=EXCLUDE_FLOATING_DEF,
    remove_duplicate_substructs=REMOVE_DUPLICATE_SUBSTRUCTS_DEF,
    params=None,
    out_dir_base=None,
    out_ext=OUT_EXT_DEF,
    db_file=None,
    overwrite=False,
    all_iters=False,
    log=None,
    num_proc=None,
    parallel_mode=None,
    verbose=False,
):
    """Generate E3FP fingerprints from SDF files."""
    setup_logging(log, verbose=verbose)

    if params is not None:
        params = read_params(params, fill_defaults=True)
        bits = get_value(params, "fingerprinting", "bits", int)
        first = get_value(params, "fingerprinting", "first", int)
        level = get_value(params, "fingerprinting", "level", int)
        radius_multiplier = get_value(
            params, "fingerprinting", "radius_multiplier", float
        )
        counts = get_value(params, "fingerprinting", "counts", bool)
        stereo = get_value(params, "fingerprinting", "stereo", bool)
        include_disconnected = get_value(
            params, "fingerprinting", "include_disconnected", bool
        )
        rdkit_invariants = get_value(
            params, "fingerprinting", "rdkit_invariants", bool
        )
        exclude_floating = get_value(
            params, "fingerprinting", "exclude_floating", bool
        )
        remove_duplicate_substructs = get_value(
            params, "fingerprinting", "remove_duplicate_substructs", bool
        )

    para = Parallelizer(num_proc=num_proc, parallel_mode=parallel_mode)

    if para.rank == 0:
        logging.info("Initializing E3FP generation.")
        logging.info("Getting SDF files")

        if len(sdf_files) == 1 and os.path.isdir(sdf_files[0]):
            from glob import glob

            sdf_files = glob("{:s}/*sdf*".format(sdf_files[0]))

        data_iterator = make_data_iterator(sdf_files)

        logging.info("SDF File Number: {:d}".format(len(sdf_files)))
        if out_dir_base is not None:
            logging.info("Out Directory Basename: {:s}".format(out_dir_base))
            logging.info("Out Extension: {:s}".format(out_ext))
        if db_file is not None:
            logging.info("Database File: {:s}".format(db_file))
        if db_file is None and out_dir_base is None:
            sys.exit("Either `db_file` or `out_dir_base` must be specified.")
        logging.info("Max First Conformers: {:d}".format(first))
        logging.info("Bits: {:d}".format(bits))
        logging.info("Level/Max Iterations: {:d}".format(level))
        logging.info(
            "Shell Radius Multiplier: {:.4g}".format(radius_multiplier)
        )
        logging.info("Stereo Mode: {!s}".format(stereo))
        if include_disconnected:
            logging.info("Connected-only mode: on")
        if rdkit_invariants:
            logging.info("Invariant type: RDKit")
        else:
            logging.info("Invariant type: Daylight")
        logging.info("Parallel Mode: {!s}".format(para.parallel_mode))
        logging.info("Starting")
    else:
        data_iterator = iter([])

    fp_kwargs = {
        "first": first,
        "bits": bits,
        "level": level,
        "radius_multiplier": radius_multiplier,
        "stereo": stereo,
        "counts": counts,
        "include_disconnected": include_disconnected,
        "rdkit_invariants": rdkit_invariants,
        "exclude_floating": exclude_floating,
        "remove_duplicate_substructs": remove_duplicate_substructs,
        "out_dir_base": out_dir_base,
        "out_ext": out_ext,
        "all_iters": all_iters,
        "overwrite": overwrite,
        "save": False,
    }
    if out_dir_base is not None:
        fp_kwargs["save"] = True

    run_kwargs = {"kwargs": fp_kwargs}

    results_iter = para.run_gen(
        fprints_dict_from_sdf, data_iterator, **run_kwargs
    )

    if db_file is not None:
        fprints = []
        for result, data in results_iter:
            try:
                fprints.extend(result.get(level, result[max(result.keys())]))
            except (AttributeError, ValueError):
                # fprinting failed, assume logged in method
                continue
        if len(fprints) > 0:
            db = FingerprintDatabase(fp_type=type(fprints[0]), level=level)
            db.add_fingerprints(fprints)
            db.savez(db_file)
            logging.info(
                (
                    "Saved FingerprintDatabase with fingerprints to " "{:s}"
                ).format(db_file)
            )
    else:
        list(results_iter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        """Generate E3FP fingerprints from SDF files.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "sdf_files",
        nargs="+",
        type=str,
        help="""Path to SDF file(s), each with one molecule
                             and multiple conformers.""",
    )
    parser.add_argument(
        "-b",
        "--bits",
        type=int,
        default=BITS,
        help="""Set number of bits for final folded
                             fingerprint. If -1 or None, unfolded (2^32-bit)
                             fingerprints are generated.""",
    )
    parser.add_argument(
        "--first",
        type=int,
        default=FIRST_DEF,
        help="""Set maximum number of first conformers for
                             which to generate fingerprints.""",
    )
    parser.add_argument(
        "-m",
        "--level",
        "--max_iterations",
        type=int,
        default=LEVEL_DEF,
        help="""Maximum number of iterations for fingerprint
                             generation. If -1, fingerprinting is run until
                             termination, and `all_iters` is set to False.""",
    )
    parser.add_argument(
        "-r",
        "--radius_multiplier",
        "--shell_radius",
        type=float,
        default=RADIUS_MULTIPLIER_DEF,
        help="""Distance to increment shell radius at around
                             each atom, starting at 0.0.""",
    )
    parser.add_argument(
        "--stereo",
        type=bool,
        default=STEREO_DEF,
        help="""Differentiate by stereochemistry.""",
    )
    parser.add_argument(
        "--counts",
        type=bool,
        default=COUNTS_DEF,
        help="""Store counts-based E3FC instead of default
                             bit-based.""",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="""INI formatted file with parameters. If
                             provided, all parameters controlling conformer
                             generation are ignored.""",
    )
    # parser.add_argument('--substruct_db', type=str, default=None,
    #                     help="""Filename to save database mapping identifiers
    #                          to substructures.""")
    # parser.add_argument('--out_format', type=str, default="E3FP",
    #                     choices=["E3FP", "RDKit"],
    #                     help="""Format of saved fingerprint.""")
    parser.add_argument(
        "-o",
        "--out_dir_base",
        type=str,
        default=None,
        help="""Basename for output directory to save
                             fingerprints. Iteration number is appended to
                             basename.""",
    )
    parser.add_argument(
        "--out_ext",
        type=str,
        default=OUT_EXT_DEF,
        choices=[".fp.pkl", ".fp.gz", ".fp.bz2"],
        help="""Extension for fingerprint pickles.""",
    )
    parser.add_argument(
        "-d",
        "--db_file",
        type=str,
        default="fingerprints.fpz",
        help="""Output file containing FingerprintDatabase
                             object""",
    )
    parser.add_argument(
        "--all_iters",
        action="store_true",
        help="""Save fingerprints from all iterations to
                             file(s).""",
    )
    parser.add_argument(
        "-O",
        "--overwrite",
        action="store_true",
        help="""Overwrite existing file(s).""",
    )
    parser.add_argument(
        "-l", "--log", type=str, default=None, help="Log filename."
    )
    parser.add_argument(
        "-p",
        "--num_proc",
        type=int,
        default=None,
        help="""Set number of processors to use.""",
    )
    parser.add_argument(
        "--parallel_mode",
        type=str,
        default=None,
        choices=list(ALL_PARALLEL_MODES),
        help="""Set parallelization mode to use.""",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Run with extra verbosity.",
    )
    params = parser.parse_args()

    kwargs = dict(params._get_kwargs())
    sdf_files = kwargs.pop("sdf_files")
    run(sdf_files, **kwargs)

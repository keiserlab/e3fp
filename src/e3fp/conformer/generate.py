"""Generate conformers from SMILES or MOL2 files.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
from __future__ import division, print_function
import sys
import os
import logging
import argparse

from rdkit.Chem import AllChem
from python_utilities.parallel import (
    Parallelizer,
    make_data_iterator,
    ALL_PARALLEL_MODES,
)
from python_utilities.scripting import setup_logging
from python_utilities.io_tools import touch_dir, HDF5Buffer
from e3fp.config.params import read_params, get_default_value, get_value
from e3fp.conformer.util import (
    mol2_generator,
    smiles_generator,
    mol_from_mol2,
    mol_from_smiles,
    mol_to_sdf,
    mol_to_standardised_mol,
)
from e3fp.conformer.generator import FORCEFIELD_CHOICES, ConformerGenerator

STANDARDISE_DEF = get_default_value("preprocessing", "standardise", bool)
NUM_CONF_DEF = get_default_value("conformer_generation", "num_conf", int)
FIRST_DEF = get_default_value("conformer_generation", "first", int)
POOL_MULTIPLIER_DEF = get_default_value(
    "conformer_generation", "pool_multiplier", int
)
RMSD_CUTOFF_DEF = get_default_value(
    "conformer_generation", "rmsd_cutoff", float
)
MAX_ENERGY_DIFF_DEF = get_default_value(
    "conformer_generation", "max_energy_diff", float
)
FORCEFIELD_DEF = get_default_value("conformer_generation", "forcefield")
SEED_DEF = get_default_value("conformer_generation", "seed", int)
OUTDIR_DEF = get_default_value("conformer_generation", "out_dir")
COMPRESS_DEF = get_default_value("conformer_generation", "compress")


def generate_conformers(
    input_mol,
    name=None,
    standardise=STANDARDISE_DEF,
    num_conf=NUM_CONF_DEF,
    first=FIRST_DEF,
    pool_multiplier=POOL_MULTIPLIER_DEF,
    rmsd_cutoff=RMSD_CUTOFF_DEF,
    max_energy_diff=MAX_ENERGY_DIFF_DEF,
    forcefield=FORCEFIELD_DEF,
    seed=SEED_DEF,
    out_file=None,
    out_dir=OUTDIR_DEF,
    save=False,
    compress=COMPRESS_DEF,
    overwrite=False,
):
    """Generate and save conformers for molecules.

    Parameters
    ----------
    input_mol : RDKit Mol
        Mol with a single conformer from which to generate conformers.
    name : str, optional
        Name of molecule.
    standardise : bool, optional
        Standardise mol before generating conformers.
    num_conf : int, optional
        If int, this is the target number of conformations. If -1, number
        of conformations is automatically chosen based on number of rotatable
        bonds.
    first : int, optional
        Number of first conformers to return. Does not impact conformer
        generator process, except may terminate conformer generation early when
        this many of conformers have been accepted.
    pool_multiplier : int, optional
        Factor to multiply by `num_conf`. The resulting number of conformations
        will be generated, then pruned to `num_conf`.
    rmsd_cutoff : float, optional
        RMSD threshold above which to accept two conformations as different
    max_energy_diff : float, optional
        Maximum energy difference between lowest energy conformer and any
        accepted conformer.
    forcefield : {'uff', 'mmff94', 'mmff94s'}, optional
        Forcefield to use for minimization of conformers.
    seed : int, optional
        Random seed for conformer generation. If -1, the random number
        generator is unseeded.
    out_file : str, optional
        Filename to save output, if `save` is True. If None, filename will be
        `name`.sdf, optionally with a compressed extension.
    out_dir : str, optional
        Directory where output files will be saved if `save` is True.
    save : bool, optional
        Save conformers to `out_file` in `out_dir`.
    compress : int, optional
        Compression of SDF files.
        None: auto. Mode is chosen based on extension, defaulting to SDF.
        0: File is not compressed.
        1: File is gzipped (.gz)
        2: File is bzipped (.bz2)
    overwrite : bool, optional
        Overwrite output files if they already exist.

    Returns
    -------
    bool
        If something went wrong, only return False. Otherwise return below.
    tuple
        Tuple with molecule name, number of rotatable bonds, numpy array of
        indices of final conformations, numpy array of energies of all
        conformations generated, and 2D numpy array of pairwise RMSDs between
        final conformations.
    """
    if name is None:
        name = input_mol.GetProp("_Name")

    if standardise:
        input_mol = mol_to_standardised_mol(input_mol)

    if save:
        if out_file is None:
            extensions = ("", ".gz", ".bz2")
            if compress not in (0, 1, 2):
                compress = 0
            out_file = os.path.join(
                out_dir, "{}.sdf{}".format(name, extensions[compress])
            )

        if os.path.exists(out_file) and not overwrite:
            logging.warning("{} already exists. Skipping.".format(out_file))
            return False

    logging.info("Generating conformers for {}.".format(name))
    try:
        conf_gen = ConformerGenerator(
            num_conf=num_conf,
            first=first,
            pool_multiplier=pool_multiplier,
            rmsd_cutoff=rmsd_cutoff,
            max_energy_diff=max_energy_diff,
            forcefield=forcefield,
            seed=seed,
            get_values=True,
        )
        mol, values = conf_gen.generate_conformers(input_mol)
        logging.info(
            "Generated {:d} conformers for {}.".format(
                mol.GetNumConformers(), name
            )
        )
    except Exception:
        logging.warning(
            "Problem generating conformers for {}.".format(name), exc_info=True
        )
        return False

    if save:
        try:
            mol_to_sdf(mol, out_file)
            logging.info(
                "Saved conformers for {} to {}.".format(name, out_file)
            )
        except Exception:
            logging.warning(
                "Problem saving conformers for {} to {}.".format(
                    name, out_file
                ),
                exc_info=True,
            )
    return (mol, name, AllChem.CalcNumRotatableBonds(mol)) + values


def values_to_hdf5(hdf5_buffer, values):
    """Write specific values to `hdf5_buffer`.

    Parameters
    ----------
    hdf5_buffer : HDF5Buffer
        HDF5 buffer to write to.
    values : tuple
        Tuple of values to be written to buffer. Values and types should be
        `name` (str), `num_rotatable_bonds` (int), `target_conformers` (int),
        `indices` (numpy long array), `energies` (numpy float array),
        and `rmsd` (numpy float array). `rmsd` should be a square array with
        size along each dimension equal to length of `indices`.

    Returns
    -------
    bool
        True if success, False if not.
    """
    try:
        (
            mol,
            name,
            num_rotatable_bonds,
            target_conformers,
            indices,
            energies,
            rmsd,
        ) = values

        group_dict = {
            "targetConfNum": {"data": target_conformers, "dtype": int},
            "numRotatableBonds": {"data": num_rotatable_bonds, "dtype": int},
            "indices": {
                "data": indices,
                "dtype": int,
                "compression": "gzip",
                "compression_opts": 9,
            },
            "energies": {
                "data": energies,
                "dtype": float,
                "compression": "gzip",
                "compression_opts": 9,
            },
            "rmsd": {
                "data": rmsd,
                "dtype": float,
                "compression": "gzip",
                "compression_opts": 9,
            },
        }

        hdf5_buffer.add_group(name, group_dict)
        logging.debug(
            "Wrote values for {} to {}.".format(name, hdf5_buffer.filename)
        )
        return True
    except:
        logging.error(
            "Problem writing values to {}.".format(hdf5_buffer.filename),
            exc_info=True,
        )
        return False


def run(
    mol2=None,
    smiles=None,
    standardise=STANDARDISE_DEF,
    num_conf=NUM_CONF_DEF,
    first=FIRST_DEF,
    pool_multiplier=POOL_MULTIPLIER_DEF,
    rmsd_cutoff=RMSD_CUTOFF_DEF,
    max_energy_diff=MAX_ENERGY_DIFF_DEF,
    forcefield=FORCEFIELD_DEF,
    seed=SEED_DEF,
    params=None,
    prioritize=False,
    out_dir=OUTDIR_DEF,
    compress=COMPRESS_DEF,
    overwrite=False,
    values_file=None,
    log=None,
    num_proc=None,
    parallel_mode=None,
    verbose=False,
):
    """Run conformer generation."""
    setup_logging(log, verbose=verbose)

    if params is not None:
        params = read_params(params)
        standardise = get_value(params, "preprocessing", "standardise", bool)
        num_conf = get_value(params, "conformer_generation", "num_conf", int)
        first = get_value(params, "conformer_generation", "first", int)
        pool_multiplier = get_value(
            params, "conformer_generation", "pool_multiplier", int
        )
        rmsd_cutoff = get_value(
            params, "conformer_generation", "rmsd_cutoff", float
        )
        max_energy_diff = get_value(
            params, "conformer_generation", "max_energy_diff", float
        )
        forcefield = get_value(params, "conformer_generation", "forcefield")
        seed = get_value(params, "conformer_generation", "seed", int)

    # check args
    if forcefield not in FORCEFIELD_CHOICES:
        raise ValueError(
            "Specified forcefield {} is not in valid options {!r}".format(
                forcefield, FORCEFIELD_CHOICES
            )
        )

    para = Parallelizer(num_proc=num_proc, parallel_mode=parallel_mode)

    # Check to make sure args make sense
    if mol2 is None and smiles is None:
        if para.is_master():
            parser.print_usage()
            logging.error("Please provide mol2 file or a SMILES file.")
        sys.exit()

    if mol2 is not None and smiles is not None:
        if para.is_master():
            parser.print_usage()
            logging.error("Please provide only a mol2 file OR a SMILES file.")
        sys.exit()

    if num_proc and num_proc < 1:
        if para.is_master():
            parser.print_usage()
            logging.error(
                "Please provide more than one processor with `--num_proc`."
            )
        sys.exit()

    # Set up input type
    if mol2 is not None:
        in_type = "mol2"
    elif smiles is not None:
        in_type = "smiles"

    if para.is_master():
        if in_type == "mol2":
            logging.info("Input type: mol2 file(s)")
            logging.info("Input file number: {:d}".format(len(mol2)))
            mol_iter = (
                mol_from_mol2(_mol2_file, _name, standardise=standardise)
                for _mol2_file, _name in mol2_generator(*mol2)
            )
        else:
            logging.info("Input type: Detected SMILES file(s)")
            logging.info("Input file number: {:d}".format(len(smiles)))
            mol_iter = (
                mol_from_smiles(_smiles, _name, standardise=standardise)
                for _smiles, _name in smiles_generator(*smiles)
            )

        if prioritize:
            logging.info(
                (
                    "Prioritizing mols with low rotatable bond number"
                    " and molecular weight first."
                )
            )
            mols_with_properties = [
                (
                    AllChem.CalcNumRotatableBonds(mol),
                    AllChem.CalcExactMolWt(mol),
                    mol,
                )
                for mol in mol_iter
                if mol is not None
            ]
            data_iterator = make_data_iterator(
                (x[-1] for x in sorted(mols_with_properties))
            )
        else:
            data_iterator = make_data_iterator(
                (x for x in mol_iter if x is not None)
            )

        # Set up parallel-specific options
        logging.info("Parallel Type: {}".format(para.parallel_mode))

        # Set other options
        touch_dir(out_dir)

        if not num_conf:
            num_conf = -1

        logging.info("Out Directory: {}".format(out_dir))
        logging.info("Overwrite Existing Files: {}".format(overwrite))
        if values_file is not None:
            if os.path.exists(values_file) and overwrite is not True:
                value_args = (values_file, "a")
                logging.info("Values file: {} (append)".format((values_file)))
            else:
                value_args = (values_file, "w")
                logging.info(
                    "Values file: {} (new file)".format((values_file))
                )
        if num_conf is None or num_conf == -1:
            logging.info("Target Conformer Number: auto")
        else:
            logging.info("Target Conformer Number: {:d}".format(num_conf))
        if first is None or first == -1:
            logging.info("First Conformers Number: all")
        else:
            logging.info("First Conformers Number: {:d}".format(first))
        logging.info("Pool Multiplier: {:d}".format(pool_multiplier))
        logging.info("RMSD Cutoff: {:.4g}".format(rmsd_cutoff))
        if max_energy_diff is None:
            logging.info("Maximum Energy Difference: None")
        else:
            logging.info(
                "Maximum Energy Difference: {:.4g} kcal".format(
                    max_energy_diff
                )
            )
        logging.info("Forcefield: {}".format(forcefield.upper()))
        if seed != -1:
            logging.info("Seed: {:d}".format(seed))

        logging.info("Starting.")
    else:
        data_iterator = iter([])

    gen_conf_kwargs = {
        "out_dir": out_dir,
        "num_conf": num_conf,
        "rmsd_cutoff": rmsd_cutoff,
        "max_energy_diff": max_energy_diff,
        "forcefield": forcefield,
        "pool_multiplier": pool_multiplier,
        "first": first,
        "seed": seed,
        "save": True,
        "overwrite": overwrite,
        "compress": compress,
    }

    run_kwargs = {"kwargs": gen_conf_kwargs}

    results_iterator = para.run_gen(
        generate_conformers, data_iterator, **run_kwargs
    )

    if para.is_master() and values_file is not None:
        hdf5_buffer = HDF5Buffer(*value_args)

    for result, data in results_iterator:
        if (
            para.is_master()
            and values_file is not None
            and result is not False
        ):
            values_to_hdf5(hdf5_buffer, result)

    if para.is_master() and values_file is not None:
        hdf5_buffer.flush()
        hdf5_buffer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Generate conformers from mol2 or SMILES",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--mol2",
        nargs="+",
        type=str,
        default=None,
        help="Path to mol2 file(s), each with one molecule.",
    )
    parser.add_argument(
        "-s",
        "--smiles",
        nargs="+",
        type=str,
        default=None,
        help="""Path to file(s) with SMILES and name. (space-
                             separated)""",
    )
    parser.add_argument(
        "--standardise",
        type=bool,
        default=STANDARDISE_DEF,
        help="""Clean molecules before generating conformers
                             by standardisation.""",
    )
    parser.add_argument(
        "-n",
        "--num_conf",
        type=int,
        default=NUM_CONF_DEF,
        help="""Set single number of conformers to use. -1
                             results in auto choosing.""",
    )
    parser.add_argument(
        "--first",
        type=int,
        default=FIRST_DEF,
        help="""Set maximum number of first conformers to
                             accept. Conformer generation is unaffected,
                             except it may terminate early when this number of
                             conformers is reached.""",
    )
    parser.add_argument(
        "--pool_multiplier",
        type=int,
        default=POOL_MULTIPLIER_DEF,
        help="""Factor to multiply `num_conf` by to generate
                             conformers. Results are then pruned to
                             `num_conf`.""",
    )
    parser.add_argument(
        "-r",
        "--rmsd_cutoff",
        type=float,
        default=RMSD_CUTOFF_DEF,
        help="Choose RMSD cutoff between conformers",
    )
    parser.add_argument(
        "-e",
        "--max_energy_diff",
        type=float,
        default=MAX_ENERGY_DIFF_DEF,
        help="""Maximum energy difference between lowest
                             energy conformer and any accepted conformer.""",
    )
    parser.add_argument(
        "-f",
        "--forcefield",
        type=str,
        choices=FORCEFIELD_CHOICES,
        default=FORCEFIELD_DEF,
        help="""Choose forcefield for minimization.""",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED_DEF,
        help="""Random seed for conformer generation.""",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default=OUTDIR_DEF,
        help="""Directory to save conformers.""",
    )
    parser.add_argument(
        "-C",
        "--compress",
        default=COMPRESS_DEF,
        type=int,
        choices={None, 0, 1, 2},
        help="""Compression to use for SDF files. None and 0
                             default to uncompressed ".sdf". 1 and 2 result in
                             gzipped and bzipped SDF files, respectively.""",
    )
    parser.add_argument(
        "-O",
        "--overwrite",
        action="store_true",
        help="""Overwrite existing conformer files.""",
    )
    parser.add_argument(
        "--values_file",
        type=str,
        default=None,
        help="""Save RMSDs and energies to specified hdf5
                             file.""",
    )
    parser.add_argument(
        "--prioritize",
        action="store_true",
        help="""Prioritize likely fast molecules first.""",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="""INI formatted file with parameters. If
                             provided, all parameters controlling conformer
                             generation are ignored.""",
    )
    parser.add_argument(
        "-l", "--log", type=str, default=None, help="Generate logfile."
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
        help="""Set number of processors to use.""",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Run with extra verbosity.",
    )
    params = parser.parse_args()

    kwargs = dict(params._get_kwargs())
    run(**kwargs)

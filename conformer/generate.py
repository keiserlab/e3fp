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

from python_utilities.parallel import Parallelizer, make_data_iterator, \
                                        ALL_PARALLEL_MODES
from python_utilities.scripting import setup_logging
from python_utilities.io_tools import touch_dir, HDF5Buffer
from e3fp.conformer.generator import ConformerGenerator
from e3fp.conformer.util import mol2_generator, smiles_generator, \
                                mol_from_mol2, mol_from_smiles, mol_to_sdf, \
                                mol_to_standardised_mol


def generate_conformers(input_mol, name=None, standardise=False, out_file=None,
                        out_dir="conformers", num_conf=-1, first=-1,
                        pool_multiplier=1, rmsd_cutoff=0.5,
                        max_energy_diff=None, forcefield='uff',
                        save=False, overwrite=False, compress=None):
    """Generate and save conformers for molecules.

    Parameters
    ----------
    data : str
        String with mol2_file or SMILES string for `in_type` of "mol2"
        or "smiles", respectively.
    out_file : str
        Output filename. If None, filename is `name`.sdf, optionally with a
        compressed extension.
    out_dir : str, optional (default "conformers")
        Directory where output files will be saved.
    num_conf : int, optional (default -1)
        If int, this is the target number of conformations. If -1, number
        of conformations is automatically chosen based on number of rotatable
        bonds.
    first : int, optional (default -1)
        Number of first conformers to return. Does not impact conformer
        generator process, except may terminate conformer generation early when
        this many of conformers have been accepted.
    pool_multiplier : int, optional (default 1)
        Factor to multiply by `num_conf`. The resulting number of conformations
        will be generated, then pruned to `num_conf`.
    rmsd_cutoff : float, optional (default 0.5)
        RMSD threshold above which to accept two conformations as different
    forcefield : {'uff', 'mmff94', 'mmff94s'}, optional (default 'uff')
        Forcefield to use for minimization of conformers.
    overwrite : bool, optional (default False)
        Overwrite output files if they already exist.
    compress : int, optional (default None)
        Compression of SDF files.
        None: auto. Mode is chosen based on extension, defaulting to SDF.
        0: File is not compressed.
        1: File is gzipped (.gz)
        2: File is bzipped (.bz2)

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
            out_file = os.path.join(out_dir,
                                    "%s.sdf%s" % (name, extensions[compress]))

        if os.path.exists(out_file) and not overwrite:
            logging.warning("%s already exists. Skipping." % (out_file))
            return False

    logging.info("Generating conformers for %s." % name)
    try:
        conf_gen = ConformerGenerator(max_conformers=num_conf,
                                      first_conformers=first,
                                      pool_multiplier=pool_multiplier,
                                      rmsd_threshold=rmsd_cutoff,
                                      max_energy_diff=max_energy_diff,
                                      force_field=forcefield,
                                      get_values=True)
        mol, values = conf_gen.generate_conformers(input_mol)
        logging.info("Generated conformers for %s." % name)
    except:
        logging.warning("Problem generating conformers for %s." % name,
                        exc_info=True)
        return False

    if save:
        try:
            mol_to_sdf(mol, out_file)
            logging.info("Saved conformers for %s to %s." % (name, out_file))
        except:
            logging.warning(
                "Problem saving conformers for %s to %s." % (name, out_file),
                exc_info=True)
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
        (name, num_rotatable_bonds,
         target_conformers, indices, energies, rmsd) = values

        group_dict = {"targetConfNum": {"data": target_conformers,
                                        "dtype": int},
                      "numRotatableBonds": {"data": num_rotatable_bonds,
                                            "dtype": int},
                      "indices": {"data": indices, "dtype": int,
                                  "compression": "gzip",
                                  "compression_opts": 9},
                      "energies": {"data": energies, "dtype": float,
                                   "compression": "gzip",
                                   "compression_opts": 9},
                      "rmsd": {"data": rmsd, "dtype": float,
                               "compression": "gzip",
                               "compression_opts": 9}
                      }

        hdf5_buffer.add_group(name, group_dict)
        logging.debug("Wrote values for %s to %s." % (name,
                                                      hdf5_buffer.filename))
        return True
    except:
        logging.error("Problem writing values to %s." % (
            hdf5_buffer.filename),
            exc_info=True)
        return False


def run(mol2=None, smiles=None, out_dir="conformers", num_conf=None,
        first=-1, pool_multiplier=1, forcefield="uff", rmsd_cutoff=0.5,
        max_energy_diff=None, standardise=False, compress=None,
        overwrite=False, values_file=None, log=None, num_proc=None,
        parallel_mode=None, verbose=False):
    """Run the script.
    """
    setup_logging(log, verbose)

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
            logging.error(
                "Please provide only a mol2 file OR a SMILES file.")
        sys.exit()

    if num_proc and num_proc < 1:
        if para.is_master():
            parser.print_usage()
            logging.error(
                "Please provide more than one processor with `--num_proc`.")
        sys.exit()

    # Set up input type
    if mol2 is not None:
        in_type = "mol2"
    elif smiles is not None:
        in_type = "smiles"

    if para.is_master():
        if in_type == "mol2":
            logging.info("Input type: mol2 file(s)")
            logging.info("Input file number: %d" % len(mol2))
            data_iterator = make_data_iterator(
                mol_from_mol2(_mol2_file, _name, standardise=standardise)
                for _mol2_file, _name in iter(mol2_generator(*mol2)))
        else:
            logging.info("Input type: Detected SMILES file(s)")
            logging.info("Input file number: %d" % len(smiles))
            data_iterator = make_data_iterator(
                mol_from_smiles(_smiles, _name, standardise=standardise)
                for _smiles, _name in iter(smiles_generator(*smiles)))



        # Set up parallel-specific options
        logging.info("Parallel Type: %s" % para.parallel_mode)

        # Set other options
        touch_dir(out_dir)

        if not num_conf:
            num_conf = -1

        logging.info("Out Directory: %s" % out_dir)
        logging.info("Overwrite Existing Files: %s" % overwrite)
        if values_file is not None:
            if os.path.exists(values_file) and overwrite is not True:
                value_args = (values_file, 'a')
                logging.info("Values file: %s (append)" % (values_file))
            else:
                value_args = (values_file, 'w')
                logging.info("Values file: %s (new file)" % (values_file))
        logging.info("Target Conformer Number: %s" % str(num_conf))
        logging.info("Pool Multiplier: %d" % pool_multiplier)
        logging.info("RMSD Cutoff: %.4g" % rmsd_cutoff)
        if max_energy_diff is None:
            logging.info("Maximum Energy Difference: None")
        else:
            logging.info(
                "Maximum Energy Difference: %.4g kcal" % (max_energy_diff))
        logging.info("Forcefield: %s" % forcefield.upper())

        logging.info("Starting.")
    else:
        data_iterator = iter([])

    gen_conf_kwargs = {"out_dir": out_dir, "num_conf": num_conf,
                       "rmsd_cutoff": rmsd_cutoff,
                       "max_energy_diff": max_energy_diff,
                       "forcefield": forcefield,
                       "pool_multiplier": pool_multiplier, "first": first,
                       "save": True, "overwrite": overwrite,
                       "compress": compress}

    run_kwargs = {"kwargs": gen_conf_kwargs}

    results_iterator = para.run_gen(generate_conformers, data_iterator,
                                    **run_kwargs)

    if para.is_master() and values_file is not None:
        hdf5_buffer = HDF5Buffer(*value_args)

    for result, data in results_iterator:
        if para.is_master() and values_file is not None:
            values_to_hdf5(hdf5_buffer, result)

    if para.is_master() and values_file is not None:
        hdf5_buffer.flush()
        hdf5_buffer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Generate conformers from mol2 or SMILES",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--mol2', nargs='+', type=str, default=None,
                        help="Path to mol2 file(s), each with one molecule.")
    parser.add_argument('-s', '--smiles', nargs='+', type=str, default=None,
                        help="""Path to file(s) with SMILES and name.
                             (space-separated)""")
    parser.add_argument('-o', '--out_dir', type=str, default="conformers",
                        help="""Directory to save conformers.""")
    parser.add_argument('-n', '--num_conf', type=int, default=-1,
                        help="""Set single number of conformers to use. -1
                             results in auto choosing.""")
    parser.add_argument('--standardise', type=bool, default=False,
                        help="""Clean molecules before generating conformers
                             by standardisation.""")
    parser.add_argument('--first', type=int, default=-1,
                        help="""Set maximum number of first conformers to
                             accept. Conformer generation is unaffected, except
                             it may terminate early when this number of
                             conformers is reached.""")
    parser.add_argument('--pool_multiplier', type=int, default=1,
                        help="""Factor to multiply `num_conf` by to
                             generate conformers. Results are then pruned
                             to `num_conf`.""")
    parser.add_argument('-f', '--forcefield', type=str,
                        choices=['uff', 'mmff94', 'mmff94s'], default='uff',
                        help="""Choose forcefield for minimization.""")
    parser.add_argument('-r', '--rmsd_cutoff', type=float, default=0.5,
                        help="Choose RMSD cutoff between conformers")
    parser.add_argument('-e', '--max_energy_diff', type=float,
                        default=None,
                        help="""Maximum energy difference between lowest energy
                             conformer and any accepted conformer.""")
    parser.add_argument('-C', '--compress', default=None, type=int,
                        choices={None, 0, 1, 2},
                        help="""Compression to use for SDF files. None and 0
                             default to uncompressed ".sdf". 1 and 2 result in
                             gzipped and bzipped SDF files, respectively.""")
    parser.add_argument('-O', '--overwrite', action="store_true",
                        help="""Overwrite existing conformer files.""")
    parser.add_argument('--values_file', type=str, default=None,
                        help="""Save RMSDs and energies to specified hdf5
                             file.""")
    parser.add_argument('-l', '--log', type=str, default=None,
                        help="Generate logfile.")
    parser.add_argument('-p', '--num_proc', type=int, default=None,
                        help="""Set number of processors to use.""")
    parser.add_argument('--parallel_mode', type=str, default=None,
                        choices=list(ALL_PARALLEL_MODES),
                        help="""Set number of processors to use.""")
    parser.add_argument('-v', '--verbose', action="store_true",
                        help="Run with extra verbosity.")
    params = parser.parse_args()

    kwargs = dict(params._get_kwargs())
    run(**kwargs)

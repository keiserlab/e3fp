"""Methods for building with random molecule subsets.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os

import numpy as np

from e3fp.sea_utils.util import lists_dicts_to_molecules, \
                                molecules_to_lists_dicts, targets_to_dict, \
                                dict_to_targets, smiles_to_dict, \
                                dict_to_smiles, \
                                filter_targets_by_molecules, \
                                targets_to_mol_lists_targets


def sample_smiles_file(smiles_file, n, sample_smiles_file=None):
    """Save a smiles file with a random `n` smiles."""
    if sample_smiles_file is None:
        basename, ext = os.path.splitext(smiles_file)
        sample_smiles_file = "%s_sample%d%s" % (basename, n, ext)
    smiles_dict = smiles_to_dict(smiles_file)

    mol_num = len(smiles_dict.keys())
    if n >= mol_num:
        return smiles_file

    rand_mol_names = np.random.choice(smiles_dict.keys(), n,
                                      replace=False)
    rand_smiles_dict = dict([(x, smiles_dict[x]) for x in rand_mol_names])

    dict_to_smiles(sample_smiles_file, rand_smiles_dict)

    return sample_smiles_file


def sample_mol_lists_files(molecules_file, targets_file, n,
                           sample_molecules_file=None,
                           sample_targets_file=None,
                           overwrite=False):
    """Save molecules/targets files with a random `n` molecules."""
    if sample_molecules_file is None:
        basename, ext = os.path.splitext(molecules_file)
        if basename.endswith(".csv"):
            basename.replace(".csv", "")
            ext = ".csv%s" % ext
        sample_molecules_file = "%s_sample%d%s" % (basename, n,
                                                   ext)

    if sample_targets_file is None:
        basename, ext = os.path.splitext(targets_file)
        if basename.endswith(".csv"):
            basename.replace(".csv", "")
            ext = ".csv%s" % ext
        sample_targets_file = "%s_sample%d%s" % (basename, n, ext)

    if overwrite or not os.path.isfile(sample_molecules_file):
        smiles_dict, mol_lists_dict, fp_type = molecules_to_lists_dicts(
            molecules_file)

        mol_num = len(mol_lists_dict.keys())
        if n >= mol_num:
            return (molecules_file, targets_file)

        rand_mol_names = np.random.choice(mol_lists_dict.keys(), n,
                                          replace=False)
        rand_smiles_dict = dict([(x, smiles_dict[x]) for x in rand_mol_names])
        rand_mol_lists_dict = dict([(x, mol_lists_dict[x])
                                    for x in rand_mol_names])
        del smiles_dict, mol_lists_dict, rand_mol_names
        lists_dicts_to_molecules(sample_molecules_file, rand_smiles_dict,
                                 rand_mol_lists_dict, fp_type)
        del rand_smiles_dict
    else:
        _, rand_mol_lists_dict, _ = molecules_to_lists_dicts(
            sample_molecules_file)
        del _

    if overwrite or not os.path.isfile(sample_targets_file):
        targets_dict = targets_to_dict(targets_file)
        rand_targets_dict = filter_targets_by_molecules(targets_dict,
                                                        rand_mol_lists_dict)
        del targets_dict
        rand_mol_lists_targets = targets_to_mol_lists_targets(
            rand_targets_dict, rand_mol_lists_dict)
        if len(rand_mol_lists_targets) > 0:
            dict_to_targets(sample_targets_file, rand_mol_lists_targets)
        else:
            dict_to_targets(sample_targets_file, rand_targets_dict)

    return (sample_molecules_file, sample_targets_file)

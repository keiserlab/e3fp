"""Functions for various pipeline use cases.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
from fpcore.fconvert import string2ascii, ascii2string

from e3fp.conformer.util import mol_from_smiles, mol_from_sdf, mol_to_sdf, \
                                mol_to_standardised_mol
from e3fp.conformer.generate import generate_conformers
from e3fp.fingerprint.fprint import Fingerprint
from e3fp.fingerprint.generate import fprints_dict_from_mol


def confs_from_smiles(smiles, name, standardise=False, first=-1,
                      confgen_kwargs={}, save=False):
    """Generate conformations of molecule from SMILES string."""
    mol = mol_from_smiles(smiles, name, standardise=standardise)
    if first != -1:
        confgen_kwargs["first"] = first
    confgen_result = generate_conformers(mol, name, save=save,
                                         **confgen_kwargs)
    mol = confgen_result[0]
    return mol


def sdf_from_smiles(smiles, name, standardise=False, first=-1,
                    confgen_kwargs={}, out_file=None, out_ext=".sdf.bz2"):
    """Generate conformations from SMILES string and save to SDF file."""
    mol = confs_from_smiles(smiles, name, standardise=standardise, first=first,
                            confgen_kwargs=confgen_kwargs, save=False)
    if out_file is None:
        out_file = name + out_ext
    mol_to_sdf(mol, out_file)


def fprints_from_fprints_dict(fprints_dict, level=-1):
    """Get fingerprint at `level` from dict of level to fingerprint."""
    fprints_list = fprints_dict.get(level,
                                    fprints_dict[max(fprints_dict.keys())])
    return fprints_list


def fold_fprints(fprints_list, fold_kwargs):
    """Fold list of fingerprints."""
    return [x.fold(**fold_kwargs) for x in fprints_list]


def fprints_from_mol(mol, standardise=False, level=-1, first=-1,
                     fprint_kwargs={}, fold_kwargs={}, save=False):
    """Generate fingerprints for all `first` conformers in mol."""
    if standardise:
        mol = mol_to_standardised_mol(mol)
    fprints_dict = fprints_dict_from_mol(mol, max_iters=level,
                                         first=first, save=save,
                                         **fprint_kwargs)
    fprints_list = fprints_from_fprints_dict(fprints_dict, level=level)
    if len(fold_kwargs) > 0:
        return fold_fprints(fprints_list, fold_kwargs)
    else:
        return fprints_list


def fprints_from_smiles(smiles, name, standardise=False, level=-1, first=-1,
                        confgen_kwargs={}, fprint_kwargs={}, fold_kwargs={},
                        save=False):
    """Generate conformers and fingerprints from a SMILES string."""
    mol = confs_from_smiles(smiles, name, standardise=standardise, first=first,
                            confgen_kwargs=confgen_kwargs, save=save)
    fprints_list = fprints_from_mol(mol, level=level, first=first,
                                    fprint_kwargs=fprint_kwargs, save=save)
    if len(fold_kwargs) > 0:
        fprints_list = [x.fold(**fold_kwargs) for x in fprints_list]
    if len(fold_kwargs) > 0:
        return fold_fprints(fprints_list, fold_kwargs)
    else:
        return fprints_list


def fprints_from_sdf(sdf_file, level=-1, first=-1, fprint_kwargs={},
                     fold_kwargs={}, save=False):
    """Generate fingerprints from conformers in an SDF file."""
    mol = mol_from_sdf(sdf_file)
    fprints_list = fprints_from_mol(mol, level=level, first=first,
                                    fprint_kwargs=fprint_kwargs, save=save)
    if len(fold_kwargs) > 0:
        fprints_list = [x.fold(**fold_kwargs) for x in fprints_list]
    if len(fold_kwargs) > 0:
        return fold_fprints(fprints_list, fold_kwargs)
    else:
        return fprints_list


def fprint_to_native_tuple(fprint):
    """Convert fingerprint to tuple with native string, name."""
    bitstring = fprint.to_bitstring()
    native = string2ascii(bitstring)
    return (native, fprint.name)


def native_tuple_to_fprint(native_tuple):
    """Convert native tuple to fingerprint."""
    native, name = native_tuple
    bitstring = ascii2string(native)
    fprint = Fingerprint.from_bitstring(bitstring)
    fprint.name = name
    return fprint


def native_tuples_from_smiles(smiles, name, standardise=False, level=-1,
                              first=-1, confgen_kwargs={}, fprint_kwargs={},
                              fold_kwargs={}, save=False):
    """Generate conformers, fprints, and native encoding from SMILES string."""
    fprints_list = fprints_from_smiles(smiles, name, standardise=standardise,
                                       level=level, first=first,
                                       confgen_kwargs=confgen_kwargs,
                                       fprint_kwargs=fprint_kwargs,
                                       fold_kwargs=fold_kwargs,
                                       save=save)
    native_tuples = list(map(fprint_to_native_tuple, fprints_list))
    return native_tuples


def native_tuples_from_sdf(sdf_file, level=-1, first=-1,
                           fprint_kwargs={}, fold_kwargs={}, save=False):
    """Fingerprint conformers from SDF file and convert to native_tuples."""
    fprints_list = fprints_from_sdf(sdf_file, level=level, first=first,
                                    fprint_kwargs=fprint_kwargs,
                                    fold_kwargs=fold_kwargs, save=save)
    native_tuples = list(map(fprint_to_native_tuple, fprints_list))
    return native_tuples

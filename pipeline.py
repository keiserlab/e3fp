"""Functions for various pipeline use cases.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
from fpcore.fconvert import string2ascii, ascii2string

from e3fp.conformer.util import mol_from_smiles, mol_from_sdf
from e3fp.conformer.generate import generate_conformers
from e3fp.fingerprint.fprint import Fingerprint
from e3fp.fingerprint.generate import fprints_dict_from_mol


def fprints_from_fprints_dict(fprints_dict, level=-1):
    """Get fingerprint at `level` from dict of level to fingerprint."""
    fprints_list = fprints_dict.get(level,
                                    fprints_dict[max(fprints_dict.keys())])
    return fprints_list


def fprints_from_smiles(smiles, name, level=-1, save=False, confgen_kwargs={},
                        fprint_kwargs={}, first=-1):
    """Generate conformers and fingerprints from a SMILES string."""
    mol = mol_from_smiles(smiles, name)
    confgen_kwargs["first"] = first
    confgen_result = generate_conformers(mol, name, save=save,
                                         **confgen_kwargs)
    mol = confgen_result[0]
    fprints_dict = fprints_dict_from_mol(mol, max_iters=level,
                                         first=first, save=save,
                                         **fprint_kwargs)
    fprints_list = fprints_from_fprints_dict(fprints_dict, level=level)
    return fprints_list


def fprints_from_sdf(sdf_file, level=-1, save=False, fprint_kwargs={},
                     first=-1):
    """Generate fingerprints from conformers in an SDF file."""
    mol = mol_from_sdf(sdf_file)
    fprints_dict = fprints_dict_from_mol(mol, save=save, max_iters=level,
                                         first=first, **fprint_kwargs)
    fprints_list = fprints_from_fprints_dict(fprints_dict, level=level)
    return fprints_list


def native_tuples_from_sdf(sdf_file, level=-1, first=-1, save=False,
                           fprint_kwargs={}, fold_kwargs={}):
    """Fingerprint conformers from SDF file and convert to native_tuples."""
    fprints_list = fprints_from_sdf(sdf_file, level=level, save=save,
                                    first=first, fprint_kwargs=fprint_kwargs)
    native_tuples = []
    for fprint in fprints_list:
        folded_fp = fprint.fold(**fold_kwargs)
        bitstring = folded_fp.to_bitstring()
        native = string2ascii(bitstring)
        native_tuples.append((native, folded_fp.name))
    return native_tuples


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


def native_tuples_from_smiles(smiles, name, level=-1, save=False,
                              confgen_kwargs={}, fprint_kwargs={},
                              first=-1, fold_kwargs={}):
    """Generate conformers, fprints, and native encoding from SMILES string."""
    fprints_list = fprints_from_smiles(smiles, name, level=level, save=save,
                                       confgen_kwargs=confgen_kwargs,
                                       fprint_kwargs=fprint_kwargs,
                                       first=first)
    native_tuples = [fprint_to_native_tuple(x.fold(**fold_kwargs))
                     for x in fprints_list]
    return native_tuples

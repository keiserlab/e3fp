"""Functions for various pipeline use cases.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
from fpcore.fconvert import string2ascii, ascii2string

from e3fp.config.params import params_to_sections_dict
from e3fp.conformer.util import mol_from_smiles, mol_from_sdf, mol_to_sdf
from e3fp.conformer.generate import generate_conformers
from e3fp.fingerprint.fprint import Fingerprint
from e3fp.fingerprint.generate import fprints_dict_from_mol


def params_to_dicts(params):
    """Get params dicts for pipeline functions from INI format params file."""
    sections_dict = params_to_sections_dict(params, auto=True)

    # preproc_params will eventually be returned separately, when there's a
    # pipeline function for protonation
    preproc_params = sections_dict.get("preprocessing", {})
    confgen_params = sections_dict.get("conformer_generation", {})
    confgen_params.update(preproc_params)
    fprint_params = sections_dict.get("fingerprinting", {})
    fold_params = sections_dict.get("folding", {})
    return confgen_params, fprint_params, fold_params


def confs_from_smiles(smiles, name, confgen_params={}, save=False):
    """Generate conformations of molecule from SMILES string."""
    mol = mol_from_smiles(smiles, name)
    confgen_result = generate_conformers(mol, name, save=save,
                                         **confgen_params)
    mol = confgen_result[0]
    return mol


def sdf_from_smiles(smiles, name, confgen_params={}, out_file=None,
                    out_ext=".sdf.bz2"):
    """Generate conformations from SMILES string and save to SDF file."""
    mol = confs_from_smiles(smiles, name, confgen_params=confgen_params,
                            save=False)
    if out_file is None:
        out_file = name + out_ext
    mol_to_sdf(mol, out_file)


def fprints_from_fprints_dict(fprints_dict, level=-1):
    """Get fingerprint at `level` from dict of level to fingerprint."""
    fprints_list = fprints_dict.get(level,
                                    fprints_dict[max(fprints_dict.keys())])
    return fprints_list


def fold_fprints(fprints_list, fold_params):
    """Fold list of fingerprints."""
    if len(fold_params) > 0:
        return [x.fold(**fold_params) for x in fprints_list]
    else:
        return fprints_list


def fprints_from_mol(mol, fprint_params={}, fold_params={}, save=False):
    """Generate fingerprints for all `first` conformers in mol."""
    fprints_dict = fprints_dict_from_mol(mol, save=save, **fprint_params)
    level = fprint_params.get("level", -1)
    fprints_list = fprints_from_fprints_dict(fprints_dict, level=level)
    return fold_fprints(fprints_list, fold_params)


def fprints_from_smiles(smiles, name, confgen_params={}, fprint_params={},
                        fold_params={}, save=False):
    """Generate conformers and fingerprints from a SMILES string."""
    if save is False and "first" not in confgen_params:
        confgen_params["first"] = fprint_params.get("first", -1)
    mol = confs_from_smiles(smiles, name, confgen_params=confgen_params,
                            save=save)
    fprints_list = fprints_from_mol(mol, fprint_params=fprint_params,
                                    save=save)
    return fold_fprints(fprints_list, fold_params)


def fprints_from_sdf(sdf_file, fprint_params={}, fold_params={}, save=False):
    """Generate fingerprints from conformers in an SDF file."""
    mol = mol_from_sdf(sdf_file)
    fprints_list = fprints_from_mol(mol, fprint_params=fprint_params,
                                    save=save)
    return fold_fprints(fprints_list, fold_params)


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


def native_tuples_from_smiles(smiles, name, confgen_params={},
                              fprint_params={}, fold_params={}, save=False):
    """Generate conformers, fprints, and native encoding from SMILES string."""
    fprints_list = fprints_from_smiles(smiles, name,
                                       confgen_params=confgen_params,
                                       fprint_params=fprint_params,
                                       fold_params=fold_params, save=save)
    native_tuples = list(map(fprint_to_native_tuple, fprints_list))
    return native_tuples


def native_tuples_from_sdf(sdf_file, fprint_params={}, fold_params={},
                           save=False):
    """Fingerprint conformers from SDF file and convert to native_tuples."""
    fprints_list = fprints_from_sdf(sdf_file, fprint_params=fprint_params,
                                    fold_params=fold_params, save=save)
    native_tuples = list(map(fprint_to_native_tuple, fprints_list))
    return native_tuples

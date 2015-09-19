"""Utilities for handling RDKit smiles, mols, and conformers.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import logging

import rdkit
import rdkit.Chem
import rdkit.Chem.PropertyMol
from rdkit.Chem.PropertyMol import PropertyMol

from python_utilities.io_tools import smart_open, touch_dir


def smiles_generator(*filenames):
    """Parse SMILES file(s) and yield (name, smile).

    Parameters
    ----------
    files : iterable object
        List of files containing smiles. File must contain one smile per
        line, followed by a space and then the molecule name.

    Yields
    ------
    tuple:
        ``tuple`` of the format (smile, name).
    """
    for filename in filenames:
        with smart_open(filename, "rb") as f:
            for i, line in enumerate(f):
                values = line.rstrip().split()
                if len(values) == 2:
                    yield tuple(values)
                else:
                    logging.warning(
                        "Line %d of %s has %d entries. Expected 2." % (
                            i + 1, filename, len(values)), exc_info=True
                    )


def mol2_generator(*filenames):
    """Parse name from mol2 filename and return generator.

    Parameters
    ----------
    files : iterable object
        List of mol2 files, where filename should be molecule name followed by
        ".mol2"

    Yields
    ------
    tuple:
        ``tuple`` of the format (file, name).
    """
    for filename in filenames:
        name = os.path.splitext(os.path.basename(filename))[0]
        yield (filename, name)


def mol_from_smiles(smile, name):
    """Generate a n RDKit PropertyMol from SMILES string.

    Parameters
    ----------
    smile : str
        SMILES string
    name : str
        Name of molecule

    Returns
    ----------
    RDKit PropertyMol : Molecule.
    """
    mol = PropertyMol(rdkit.Chem.MolFromSmiles(smile))
    mol.SetProp("_Name", name)
    mol.SetProp("_SMILES", smile)
    return mol


def mol_from_mol2(mol2_file, name=None):
    """Read a mol2 file into an RDKit PropertyMol.

    Parameters
    ----------
    mol2_file : str
        path to a MOL2 file
    name : str, optional (default: None)
        Name of molecule. If not provided, uses file basename as name

    Returns
    ----------
    RDKit PropertyMol : Molecule.
    """
    if name is None:
        name = os.path.splitext(os.path.basename(mol2_file))[0]
    mol = PropertyMol(rdkit.Chem.MolFromMol2File(mol2_file))
    mol.SetProp("_Name", name)
    return mol


def mol_from_sdf(sdf_file):
    """Read SDF file into an RDKit ``Mol`` object.

    Parameters
    ----------
    sdf_file : str
        Path to an SDF file

    Returns
    -------
    RDKit Mol : ``Mol`` object with each molecule in SDF file as a conformer
    """
    mol = None
    with smart_open(sdf_file, "rb") as f:
        supplier = rdkit.Chem.ForwardSDMolSupplier(f)
        i = 0
        while True:
            try:
                new_mol = supplier.next()
            except StopIteration:
                logging.debug(
                    "Read %d conformers from %s." % (i, sdf_file))
                break
            if mol is None:
                mol = rdkit.Chem.Mol(new_mol)
                mol.RemoveAllConformers()
            conf = new_mol.GetConformers()[0]
            mol.AddConformer(conf, assignId=True)
            i += 1
    return mol


def mol_to_sdf(mol, out_file):
    """Write RDKit ``Mol`` objects to an SDF file.

    Parameters
    ----------
    mol : RDKit Mol
        A molecule containing 1 or more conformations to write to file.
    out_file : str
        Path to save SDF file.
    """
    touch_dir(os.path.dirname(out_file))
    with smart_open(out_file, "wb") as fobj:
        writer = rdkit.Chem.SDWriter(fobj)
        conf_ids = [conf.GetId() for conf in mol.GetConformers()]
        for i in conf_ids:
            writer.write(mol, confId=i)
        writer.close()
    logging.debug("Saved %d conformers to %s." % (i + 1, out_file))

"""Utilities for handling SMILES strings and RDKit mols and conformers.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import re
import copy
import logging
from collections import namedtuple

import rdkit
import rdkit.Chem
import rdkit.Chem.PropertyMol
from rdkit.Chem.PropertyMol import PropertyMol
from python_utilities.io_tools import smart_open, touch_dir


PROTO_NAME_DELIM = "-"
CONF_NAME_DELIM = "_"
MOL_ITEM_REGEX = re.compile(
    r"(?P<{0}>.+?)(?:{1}(?P<{2}>\d+))?(?:{3}(?P<{4}>\d+))?$".format(
        "mol_name",
        PROTO_NAME_DELIM,
        "proto_state_num",
        CONF_NAME_DELIM,
        "conf_num",
    )
)
MOL_ITEM_FIELDS = ("mol_name", "proto_state_num", "conf_num")
CONF_ENERGIES_PROPNAME = "_ConfEnergies"
CONF_ENERGIES_DELIM = "|"
CONF_ENERGY_PROPNAME = "Energy"

MolItemTuple = namedtuple(
    "MolItemTuple", ["mol_name", "proto_state_num", "conf_num"]
)


class MolItemName(object):
    """Class for parsing mol item names and converting to various formats."""

    def __init__(
        self,
        mol_name=None,
        proto_state_num=None,
        conf_num=None,
        proto_delim=PROTO_NAME_DELIM,
        conf_delim=CONF_NAME_DELIM,
    ):
        self.mol_name = mol_name
        self.proto_state_num = proto_state_num
        self.conf_num = conf_num
        self.proto_delim = proto_delim
        self.conf_delim = conf_delim

    @classmethod
    def from_str(
        cls,
        mol_item_name,
        mol_item_regex=MOL_ITEM_REGEX,
        mol_item_fields=MOL_ITEM_FIELDS,
        **kwargs
    ):
        fields = cls.mol_item_name_to_dict(
            mol_item_name,
            mol_item_regex=mol_item_regex,
            mol_item_fields=mol_item_fields,
        )
        return cls(
            fields["mol_name"],
            fields["proto_state_num"],
            fields["conf_num"],
            **kwargs
        )

    def to_str(self):
        return self.mol_item_name

    @classmethod
    def from_tuple(cls, fields_tuple):
        return cls(*fields_tuple)

    def to_tuple(self):
        return MolItemTuple(self.mol_name, self.proto_state_num, self.conf_num)

    @property
    def mol_name(self):
        return self._mol_name

    @mol_name.setter
    def mol_name(self, mol_name):
        self._mol_name = mol_name

    def to_mol_name(self, as_proto=False):
        if as_proto:
            return self.proto_name
        else:
            return self.mol_name

    @property
    def proto_name(self):
        return self.to_proto_name(self.proto_state_num)

    def to_proto_name(
        self, proto_state_num=None, proto_delim=PROTO_NAME_DELIM
    ):
        if proto_state_num is not None:
            return "{}{}{:d}".format(
                self.mol_name, proto_delim, proto_state_num
            )
        else:
            return self.mol_name

    @property
    def conf_name(self):
        return self.to_conf_name(conf_num=self.conf_num)

    def to_conf_name(self, conf_num=None, conf_delim=CONF_NAME_DELIM):
        if conf_num is not None:
            return "{}{}{:d}".format(self.proto_name, conf_delim, conf_num)
        else:
            return self.proto_name

    @property
    def mol_item_name(self):
        return self.conf_name

    @staticmethod
    def mol_item_name_to_dict(
        mol_item_name,
        mol_item_regex=MOL_ITEM_REGEX,
        mol_item_fields=MOL_ITEM_FIELDS,
    ):
        match = re.match(mol_item_regex, mol_item_name)
        groups = match.groups()
        fields = dict(zip(mol_item_fields, groups))
        proto_state_num = fields.get("proto_state_num")
        if proto_state_num is not None:
            fields["proto_state_num"] = int(proto_state_num)
        conf_num = fields.get("conf_num")
        if conf_num is not None:
            fields["conf_num"] = int(conf_num)
        return fields

    def copy(self):
        return copy.copy(self)

    def __repr__(self):
        return (
            "MolItemName(mol_name={}, proto_state_num={}, "
            "conf_num={})".format(
                self.mol_name, self.proto_state_num, self.conf_num
            )
        )

    def __str__(self):
        return self.conf_name

    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __gt__(self, other):
        return self.to_tuple().__gt__(other.to_tuple())

    def __lt__(self, other):
        return self.to_tuple().__lt__(other.to_tuple())

    def __hash__(self):
        return hash(self.to_tuple())


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
        `tuple` of the format (smile, name).
    """
    for filename in filenames:
        with smart_open(filename, "r") as f:
            for i, line in enumerate(f):
                values = line.rstrip("\r\n").split()
                if len(values) >= 2:
                    yield tuple(values[:2])
                else:
                    logging.warning(
                        (
                            "Line {:d} of {} has {:d} entries. Expected at least"
                            " 2.".format(i + 1, filename, len(values))
                        ),
                        exc_info=True,
                    )


def smiles_to_dict(smiles_file, unique=False, has_header=False):
    """Read SMILES file to dict."""
    smiles_gen = smiles_generator(smiles_file)
    if has_header:
        header = next(smiles_gen)
        logging.info("Skipping first (header) values: {!r}".format(header))
    if unique:
        used_smiles = set()
        smiles_dict = {}
        for smiles, name in smiles_gen:
            if name not in smiles_dict and smiles not in used_smiles:
                smiles_dict[name] = smiles
                used_smiles.add(smiles)
    else:
        smiles_dict = {name: smiles for smiles, name in smiles_gen}
    return smiles_dict


def dict_to_smiles(smiles_file, smiles_dict):
    """Write SMILES dict to file."""
    iter_to_smiles(smiles_file, sorted(smiles_dict.items()))


def iter_to_smiles(smiles_file, smiles_iter):
    """Write iterator of (mol_name, SMILES) to file."""
    with smart_open(smiles_file, "w") as f:
        for mol_name, smiles in smiles_iter:
            f.write("{} {}\n".format(smiles, mol_name))


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
        `tuple` of the format (file, name).
    """
    for filename in filenames:
        name = os.path.splitext(os.path.basename(filename))[0]
        yield (filename, name)


def mol_from_smiles(smiles, name, standardise=False):
    """Generate a n RDKit `PropertyMol` from SMILES string.

    Parameters
    ----------
    smile : str
        SMILES string
    name : str
        Name of molecule
    standardise : bool
        Clean Mol through standardisation

    Returns
    -------
    RDKit PropertyMol : Molecule.
    """
    mol = rdkit.Chem.MolFromSmiles(smiles)
    if mol is None:
        logging.error(
            "Mol creation failed from SMILES: {!r}".format((smiles, name))
        )
        return None
    if standardise:
        mol = mol_to_standardised_mol(mol, name)
    mol = PropertyMol(mol)
    mol.SetProp("_Name", name)
    mol.SetProp("_SMILES", smiles)
    return mol


def mol_from_mol2(mol2_file, name=None, standardise=False):
    """Read a mol2 file into an RDKit `PropertyMol`.

    Parameters
    ----------
    mol2_file : str
        path to a mol2 file
    name : str, optional
        Name of molecule. If not provided, uses file basename as name
    standardise : bool
        Clean mol through standardisation

    Returns
    -------
    RDKit PropertyMol : Molecule.
    """
    if name is None:
        name = os.path.splitext(os.path.basename(mol2_file))[0]
    mol = rdkit.Chem.MolFromMol2File(mol2_file)
    if standardise:
        mol = mol_to_standardised_mol(mol, name)
    mol = PropertyMol(mol)
    mol.SetProp("_Name", name)
    return mol


def mol_from_sdf(sdf_file, conf_num=None, standardise=False):
    """Read SDF file into an RDKit `Mol` object.

    Parameters
    ----------
    sdf_file : str
        Path to an SDF file
    conf_num : int or None, optional
        Maximum number of conformers to read from file. Defaults to all.
    standardise : bool (default False)
        Clean mol through standardisation

    Returns
    -------
    RDKit Mol : `Mol` object with each molecule in SDF file as a conformer
    """
    mol = None
    conf_energies = []
    with smart_open(sdf_file, "r") as f:
        supplier = rdkit.Chem.ForwardSDMolSupplier(f)
        i = 0
        while True:
            if i == conf_num:
                break
            try:
                new_mol = next(supplier)
            except StopIteration:
                logging.debug(
                    "Read {:d} conformers from {}.".format(i, sdf_file)
                )
                break

            if new_mol.HasProp(CONF_ENERGY_PROPNAME):
                conf_energies.append(
                    float(new_mol.GetProp(CONF_ENERGY_PROPNAME))
                )

            if mol is None:
                mol = rdkit.Chem.Mol(new_mol)
                mol.RemoveAllConformers()
            conf = new_mol.GetConformers()[0]
            mol.AddConformer(conf, assignId=True)
            i += 1
    if standardise:
        mol = mol_to_standardised_mol(mol)
    try:
        mol.GetProp("_Name")
    except KeyError:
        name = os.path.basename(sdf_file).split(".sdf")[0]
        mol.SetProp("_Name", name)

    if len(conf_energies) > 0:
        add_conformer_energies_to_mol(mol, conf_energies)
        mol.ClearProp(CONF_ENERGY_PROPNAME)

    return mol


def mol_to_sdf(mol, out_file, conf_num=None):
    """Write RDKit `Mol` objects to an SDF file.

    Parameters
    ----------
    mol : RDKit Mol
        A molecule containing 1 or more conformations to write to file.
    out_file : str
        Path to save SDF file.
    conf_num : int or None, optional
        Maximum number of conformers to save to file. Defaults to all.
    """
    touch_dir(os.path.dirname(out_file))
    with smart_open(out_file, "w") as fobj:
        writer = rdkit.Chem.SDWriter(fobj)
        conf_ids = [conf.GetId() for conf in mol.GetConformers()]
        conf_energies = get_conformer_energies_from_mol(mol)
        mol.ClearProp(CONF_ENERGIES_PROPNAME)
        for i in conf_ids:
            if conf_num not in {-1, None} and i >= conf_num:
                break
            try:
                conf_energy = conf_energies[i]
                mol.SetProp(CONF_ENERGY_PROPNAME, "{:.4f}".format(conf_energy))
            except (IndexError, TypeError):
                pass
            writer.write(mol, confId=i)
        writer.close()
        mol.ClearProp(CONF_ENERGY_PROPNAME)
        if conf_energies is not None:
            add_conformer_energies_to_mol(mol, conf_energies)
    logging.debug("Saved {:d} conformers to {}.".format(i + 1, out_file))


def mol_to_standardised_mol(mol, name=None):
    """Standardise mol(s)."""
    try:
        from standardiser import standardise
        from standardiser.utils import StandardiseException
    except ImportError:
        logging.warning(
            "standardiser module unavailable. Using unstandardised mol."
        )
        return mol

    if name is None:
        try:
            name = mol.GetProp("_Name")
        except KeyError:
            name = repr(mol)

    if isinstance(mol, PropertyMol):
        mol_type = PropertyMol
        mol = rdkit.Chem.Mol(mol)
    else:
        mol_type = rdkit.Chem.Mol

    logging.debug("Standardising {}".format(name))
    try:
        std_mol = standardise.run(mol)
    except AttributeError:  # backwards-compatible with old standardiser
        std_mol = standardise.apply(mol)
    except StandardiseException:
        logging.error(
            (
                "Standardisation of {} failed. Using unstandardised "
                "mol.".format(name)
            ),
            exc_info=True,
        )
        return mol_type(mol)

    std_mol = mol_type(std_mol)
    try:
        std_mol.SetProp("_Name", mol.GetProp("_Name"))
    except KeyError:
        pass

    return std_mol


def add_conformer_energies_to_mol(mol, energies):
    """Add conformer energies as mol property.

    See discussion at https://sourceforge.net/p/rdkit/mailman/message/27547551/
    """
    energies_str = CONF_ENERGIES_DELIM.join(
        "{:.4f}".format(e) for e in energies
    )
    mol.SetProp(CONF_ENERGIES_PROPNAME, energies_str)
    return mol


def get_conformer_energies_from_mol(mol):
    """Get conformer energies from mol."""
    if not mol.HasProp(CONF_ENERGIES_PROPNAME):
        return None
    energies_str = mol.GetProp(CONF_ENERGIES_PROPNAME)
    energies = [float(x) for x in energies_str.split(CONF_ENERGIES_DELIM)]
    return energies

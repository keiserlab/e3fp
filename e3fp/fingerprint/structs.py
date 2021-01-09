"""Class for defining 3D atom environments.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
from __future__ import division, print_function
from functools import reduce

import numpy as np
import rdkit.Chem

from python_utilities.io_tools import smart_open
from e3fp.fingerprint import array_ops


PDB_LINE = (
    "HETATM{atom_id:>5d} {name:<4s} LIG A   1    "
    "{coord[0]:>8.3f}{coord[1]:>8.3f}{coord[2]:>8.3f}"
    "{occupancy:>6.2f}{temp:>6.2f}          {elem:>2s}{charge:>2s}"
)


class Shell(object):
    """A container for other Shells centered on an atom.

    Shells represent all atoms explicitly within a container. Atoms are
    represented by their ids. If atoms are provided instead of shells, they
    are converted to single-atom shells. A Substruct is generated from a Shell
    on the fly by recursion through member shells. An optional identifier may
    be set.
    """

    def __init__(
        self,
        center_atom,
        shells=set(),
        radius=None,
        last_shell=None,
        identifier=None,
    ):
        if isinstance(center_atom, rdkit.Chem.Atom):
            center_atom = center_atom.GetIdx()
        elif not isinstance(center_atom, (int, np.integer)):
            raise TypeError("center_atom must be Atom or atom id")
        self._center_atom = center_atom

        self._shells = set()
        for shell in shells:
            if isinstance(shell, int):
                shell = Shell(shell)
            elif isinstance(shell, rdkit.Chem.Atom):
                shell = Shell(shell.GetIdx())
            elif not isinstance(shell, Shell):
                raise TypeError("shells must be Shells, Atoms, or atom ids")
            if shell.center_atom == self.center_atom:
                raise FormatError(
                    "member shells cannot be centered on same "
                    "center_atom as new shell"
                )
            self._shells.add(shell)
        self._shells = frozenset(self._shells)

        self.radius = radius
        self.last_shell = last_shell
        self.atoms = None
        self.substruct = None
        self.identifier = identifier
        self.is_duplicate = False
        self.duplicate = None

    @classmethod
    def from_substruct(cls, substruct):
        """Create shell with one shell for each atom in the substruct."""
        if substruct.center_atom is None:
            raise FormatError(
                "Can only create Shell from Substruct if "
                "center_atom is defined"
            )
        atoms = substruct.atoms ^ {substruct.center_atom}
        return cls(substruct.center_atom, [Shell(x) for x in atoms])

    @property
    def center_atom(self):
        return self._center_atom

    @property
    def shells(self):
        return self._shells

    @property
    def atoms(self):
        """Get all atoms explicitly within the shell."""
        if self._atoms is None:
            self._atoms = set([self.center_atom,])
            self._atoms.update([x.center_atom for x in self.shells])
        return self._atoms

    @atoms.setter
    def atoms(self, atoms):
        self._atoms = atoms

    @property
    def substruct(self):
        """Get substruct with all atoms implicitly within the shell."""
        if self._substruct is None:
            atom_sets = [set(x.substruct.atoms) for x in self.shells]
            if len(atom_sets) > 0:
                atoms = reduce(set.union, atom_sets)
            else:
                atoms = set()
            self._substruct = Substruct(
                center_atom=self.center_atom, atoms=atoms
            )
            self._substruct.shell = self
        return self._substruct

    @substruct.setter
    def substruct(self, substruct):
        if not isinstance(substruct, Substruct) and substruct is not None:
            raise TypeError("substruct must be of type Substruct")
        self._substruct = substruct

    def __repr__(self):
        return (
            "Shell(center_atom={!r}, shells={!r}, radius={!r}, "
            "last_shell={!r}, identifier={!r})"
        ).format(
            self.center_atom,
            tuple(self.shells),
            self.radius,
            self.last_shell,
            self.identifier,
        )

    def __str__(self):
        return (
            "Shell(center_atom={!r}, atoms={!r}, radius={!r}, "
            "identifier={!r})"
        ).format(
            self.center_atom, tuple(self.atoms), self.radius, self.identifier
        )

    def __hash__(self):
        return hash((self.center_atom, self.shells))

    def __eq__(self, other):
        return (self.center_atom == other.center_atom) and (
            self.shells == other.shells
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return 1 + len(self.shells)

    def __contains__(self, key):
        if isinstance(key, (int, rdkit.Chem.Atom)):
            key = Shell(key)
        return key in self.shells or key == self


class Substruct(object):
    """A container for atoms optionally centered on an atom.

    A Substruct represents all atoms implicitly within a Shell. Two Substructs
    are equal if they contain the same atoms.
    """

    def __init__(self, center_atom=None, atoms=set()):
        self.center_atom = center_atom
        self.shell = None
        self._atoms = set()
        for atom in atoms:
            if isinstance(atom, rdkit.Chem.Atom):
                atom = atom.GetIdx()
            elif not isinstance(atom, (int, np.integer)):
                raise TypeError("atoms must be Atom or atom id")
            self._atoms.add(atom)
        if self.center_atom is not None:
            self._atoms.add(self.center_atom)
        self._atoms = frozenset(self._atoms)
        self.transform_matrix = np.identity(4, dtype=np.float)

    @classmethod
    def from_shell(cls, shell):
        return shell.substruct

    @property
    def center_atom(self):
        return self._center_atom

    @center_atom.setter
    def center_atom(self, center_atom):
        if isinstance(center_atom, rdkit.Chem.Atom):
            center_atom = center_atom.GetIdx()
        elif (
            not isinstance(center_atom, (int, np.integer))
            and center_atom is not None
        ):
            raise TypeError("center_atom must be Atom or atom id")
        self._center_atom = center_atom

    @property
    def atoms(self):
        return self._atoms

    def __repr__(self):
        return "Substruct(center_atom={!r}, atoms={!r})".format(
            self.center_atom, tuple(self.atoms)
        )

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.atoms)

    def __eq__(self, other):
        return self.atoms == other.atoms

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self.atoms)

    def __contains__(self, key):
        if isinstance(key, rdkit.Chem.Atom):
            key = key.GetIdx()
        return key in self.atoms


class FormatError(Exception):
    pass


# methods/classes for shell i/o
def shell_to_pdb(
    mol, shell, atom_coords, bound_atoms_dict, out_file=None, reorient=True
):
    """Append substructure within shell to PDB.

    Parameters
    ----------
    mol : RDKit Mol
        Input mol
    shell : Shell
        A shell
    atom_coords : dict
        Dict matching atom id to coordinates.
    bound_atoms_dict : dict
        Dict matching atom id to id of bound atoms.
    out_file : str or None, optional
        File to which to append coordinates.
    reorient : bool, optional
        Use the transformation matrix in the shell to align by the stereo
        quadrants. If no transformation matrix present, centers the center
        atom.

    Returns
    -------
    list of str: list of PDB file lines, if `out_file` not specified
    """
    remark = "REMARK 400"
    header_lines = [remark + " COMPOUND", remark + " " + mol.GetProp("_Name")]
    lines = header_lines + [
        "MODEL",
    ]
    atom_ids = sorted(shell.substruct.atoms)
    atoms = [mol.GetAtomWithIdx(int(x)) for x in atom_ids]
    coords = np.asarray(list(map(atom_coords.get, atom_ids)), dtype=np.float64)
    if reorient:
        try:
            coords = array_ops.transform_array(shell.transform_matrix, coords)
        except AttributeError:
            coords -= atom_coords[shell.center_atom]

    for i, atom_id in enumerate(atom_ids):
        elem = atoms[i].GetSymbol()
        name = "{}{:d}".format(elem, atom_id + 1)
        charge = atoms[i].GetFormalCharge()
        if charge > 0:
            charge = "{:d}+".format(charge)
        elif charge < 0:
            charge = "{:d}-".format(abs(charge))
        else:
            charge = ""
        if atom_id == shell.center_atom:
            temp = 1.0
        elif atom_id in shell.atoms:
            temp = 0.5
        else:
            temp = 0.0
        pdb_entries = {
            "atom_id": atom_id,
            "name": name,
            "coord": coords[i, :].flatten(),
            "occupancy": 0.0,
            "temp": temp,
            "elem": elem,
            "charge": charge,
        }
        lines.append(PDB_LINE.format(**pdb_entries))

    # PLACEHOLDER FOR WRITING BONDS TO PDB
    # used_bonds = set()
    # write_bonds = []
    # for atom_id in atom_ids:
    #     write_bonds.append(atom_id)
    #     bound_atom_ids = bound_atoms_dict.get(atom_id, set())
    #     for bound_atom_id in bound_atom_ids:
    #         if (atom_id, bound_atom_id) in used_bonds:
    #             continue
    #         if len(write_bonds) > 3:
    #             lines.append("CONECT "+" ".join(map(str, write_bonds)))
    #             write_bonds = [atom_id,]
    #         write_bonds.append(bound_atom_id)
    #         used_bonds.add((atom_id, bound_atom_id))
    #         used_bonds.add((bound_atom_id, atom_id))

    #     lines.append("CONECT "+" ".join(map(str, write_bonds)))
    #     write_bonds = []

    lines.append("ENDMDL")

    if out_file is not None:
        with smart_open(out_file, "a") as f:
            for line in lines:
                f.write(line + "\n")
    else:
        return lines

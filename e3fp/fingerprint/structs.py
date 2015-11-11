"""Class for defining 3D atom environments.

Author: Seth Axen
E-mail: seth.axen@gmail.com"""
import numpy as np
import rdkit.Chem


class Shell(object):

    """A container for other Shells centered on an atom.

    Shells represent all atoms explicitly within a container. Atoms are
    represented by their ids. If atoms are provided instead of shells, they
    are converted to single-atom shells. A Substruct is generated from a Shell
    on the fly by recursion through member shells. An optional identifier may
    be set."""

    def __init__(self, center_atom, shells=set(), radius=None,
                 last_shell=None, identifier=None):
        if isinstance(center_atom, rdkit.Chem.Atom):
            center_atom = center_atom.GetIdx()
        elif not isinstance(center_atom, int):
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
                    "member shells cannot be centered on same center_atom as new shell")
            self._shells.add(shell)
        self._shells = frozenset(self._shells)

        self.radius = radius
        self.last_shell = last_shell
        self.atoms = None
        self.substruct = None
        self.identifier = identifier

    @classmethod
    def from_substruct(cls, substruct):
        """Create shell with one shell for each atom in the substruct."""
        if substruct.center_atom is None:
            raise FormatError(
                "Can only create Shell from Substruct if center_atom is defined")
        atoms = substruct.atoms ^ {substruct.center_atom}
        return cls(substruct.center_atom, map(Shell, atoms))

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
            self._atoms = set([self.center_atom, ])
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
                center_atom=self.center_atom,
                atoms=atoms)
        return self._substruct

    @substruct.setter
    def substruct(self, substruct):
        if not isinstance(substruct, Substruct) and substruct is not None:
            raise TypeError("substruct must be of type Substruct")
        self._substruct = substruct

    def __repr__(self):
        return ('Shell(center_atom={!r}, shells={!r}, radius={!r}, '
                'last_shell={!r}, identifier={!r})').format(
                    self.center_atom, tuple(self.shells), self.radius,
                    self.last_shell, self.identifier)

    def __str__(self):
        return ('Shell(center_atom={!r}, atoms={!r}, radius={!r}, '
                'identifier={!r})').format(
                    self.center_atom, tuple(self.atoms), self.radius,
                    self.identifier)

    def __hash__(self):
        return hash((self.center_atom, self.shells))

    def __eq__(self, other):
        return ((self.center_atom == other.center_atom) and
                (self.shells == other.shells))

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

    A Substruct represents all atoms implicitly within a Shell. Two
    Substructs are equal if they contain the same atoms."""

    def __init__(self, center_atom=None, atoms=set()):
        self.center_atom = center_atom
        self._atoms = set()
        for atom in atoms:
            if isinstance(atom, rdkit.Chem.Atom):
                atom = atom.GetIdx()
            elif not isinstance(atom, int):
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
        elif not isinstance(center_atom, int) and center_atom is not None:
            raise TypeError("center_atom must be Atom or atom id")
        self._center_atom = center_atom

    @property
    def atoms(self):
        return self._atoms

    def __repr__(self):
        return "Substruct(center_atom={!r}, atoms={!r})".format(
            self.center_atom, tuple(self.atoms))

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.atoms)

    def __eq__(self, other):
        return (self.atoms == other.atoms)

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

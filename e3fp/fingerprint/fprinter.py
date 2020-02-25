"""Tools for generating E3FP fingerprints.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
from __future__ import division, print_function
import os
import logging

import numpy as np
from rdkit import Chem
import mmh3
from python_utilities.io_tools import touch_dir
from python_utilities.scripting import setup_logging
from ..config.params import get_default_value
from .fprint import Fingerprint, CountFingerprint
from .structs import Shell, shell_to_pdb
from . import array_ops


BITS = 2 ** 32
LEVEL_DEF = get_default_value("fingerprinting", "level", int)
RADIUS_MULTIPLIER_DEF = get_default_value(
    "fingerprinting", "radius_multiplier", float
)
COUNTS_DEF = get_default_value("fingerprinting", "counts", bool)
STEREO_DEF = get_default_value("fingerprinting", "stereo", bool)
INCLUDE_DISCONNECTED_DEF = get_default_value(
    "fingerprinting", "include_disconnected", bool
)
RDKIT_INVARIANTS_DEF = get_default_value(
    "fingerprinting", "rdkit_invariants", bool
)
EXCLUDE_FLOATING_DEF = get_default_value(
    "fingerprinting", "exclude_floating", bool
)
REMOVE_DUPLICATE_SUBSTRUCTS_DEF = get_default_value(
    "fingerprinting", "remove_duplicate_substructs", bool
)
IDENT_DTYPE = np.int64  # np.dtype to use for identifiers
Y_AXIS_PRECISION = 0.1  # angstroms
Z_AXIS_PRECISION = 0.01  # rad
POLAR_CONE_RAD = np.pi / 36  # rad
MMH3_SEED = 0
BOND_TYPES = {
    None: 5,
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}

setup_logging(reset=False)


class Fingerprinter(object):
    """E3FP fingerprint generator.

    Parameters
    ----------
    bits : int or None, optional
        Maximum number of bits to which to fold returned fingerprint. Multiple
        of 2 is strongly recommended.
    level : int or None, optional
        Maximum number of iterations for fingerprint generation. If None or
        -1, run until no new substructures are identified. Because this could
        produce a different final level number for each conformer, it is
        recommended to manually specify a level.
    radius_multiplier : float, optional
        Multiple by which to increase shell size. At iteration 0, shell radius
        is 0*`radius_multiplier`, at iteration 2, radius is
        2*`radius_multiplier`, etc.
    counts : bool, optional
        Instead of simple bit-based `Fingerprint` object, generate
        `CountFingerprint` that tracks number of times each bit appears in a
        fingerprint.
    stereo : bool, optional
        Differentiate based on stereography. Resulting fingerprints are not
        comparable to non-stereo fingerprints.
    remove_duplicate_substructs : bool, optional
        If a substructure arises that corresponds to an identifier already in
        the fingerprint, then the identifier for the duplicate substructure is
        not added to fingerprint.
    include_disconnected : bool, optional:
        Include disconnected atoms from hashes and substructure. E3FP's
        advantage over ECFP relies on disconnected atoms, so the option to
        turn this off is present only for testing/comparison.
    rdkit_invariants : bool, optional
        Use the atom invariants used by RDKit for its Morgan fingerprint.
    exclude_floating : bool, optional:
        Mask atoms with no bonds (usually floating ions) from the fingerprint.
        These are often placed arbitrarily and can confound the fingerprint.

    Attributes
    ----------
    current_level : int
        The maximum level/iteration to which the fingerprinter has been run
        on the current conformer.
    level_shells : dict
        Dict matching level to set of all shells accepted at that level.
    """

    def __init__(
        self,
        bits=BITS,
        level=LEVEL_DEF,
        radius_multiplier=RADIUS_MULTIPLIER_DEF,
        stereo=STEREO_DEF,
        counts=COUNTS_DEF,
        include_disconnected=INCLUDE_DISCONNECTED_DEF,
        rdkit_invariants=RDKIT_INVARIANTS_DEF,
        exclude_floating=EXCLUDE_FLOATING_DEF,
        remove_duplicate_substructs=REMOVE_DUPLICATE_SUBSTRUCTS_DEF,
    ):
        """Initialize fingerprinter settings."""
        self.mol = None
        if level is None:
            level = -1
        self.level = level
        if not np.log2(bits).is_integer():
            logging.warning(
                "bits are not multiple of 2. Multiples of 2 are "
                "highly recommended"
            )
        self.bits = bits
        self.radius_multiplier = radius_multiplier
        if counts:
            self.fp_type = CountFingerprint
        else:
            self.fp_type = Fingerprint
        self.stereo = stereo
        self.remove_duplicate_substructs = remove_duplicate_substructs

        if self.level == -1 and not self.remove_duplicate_substructs:
            raise Exception(
                "No termination condition specified. 'level' must be "
                "provided or 'remove_duplicate_substructs' must be True"
            )

        self.include_disconnected = include_disconnected
        self.rdkit_invariants = rdkit_invariants
        self.exclude_floating = exclude_floating

        self.bond_types = BOND_TYPES
        self.reset()

    def run(self, conf=None, mol=None, return_substruct=False):
        """Generate fingerprint from provided conformer or mol and conf id.

        Parameters
        ----------
        conf : RDKit Conformer or int, optional
            Input conformer or conformer in `mol`.
        mol : RDKit Mol, optional
            Input molecule object, with at least one conformer. If `conf` not
            specified, first conformer is used.
        return_substruct : bool, optional
            Return dict mapping substructure to fingerprint indices. Keys are
            indices, values are list of substructures, represented as a tuple
            of atom indices where the first index is the central atom and the
            remaining indices (within the sphere) are sorted.
        """
        if mol is None:  # mol not provided; get from conf
            try:
                mol = conf.GetOwningMol()
            except AttributeError:  # conf is int ID; use existing mol
                mol = self.mol
        else:
            if not isinstance(conf, Chem.Conformer):
                try:
                    conf = mol.GetConformer(conf)
                except TypeError:  # conf isn't ID either. Fall back to first
                    conf = mol.GetConformer(0)

        if mol is not self.mol:
            self.reset_mol()
            self.initialize_mol(mol)
        elif conf is not self.conf:
            self.reset_conf()

        self.initialize_conformer(conf)

        for i in iter(self):
            pass

    def reset(self):
        """Clear all variables associated with the last run."""
        self.reset_mol()

    def reset_mol(self):
        """Clear all variables associated with the molecule."""
        self.atoms = None
        self.bound_atoms_dict = {}
        self.connectivity = {}
        self.init_identifiers = {}
        self.reset_conf()

    def reset_conf(self):
        """Clear only conformer-specific variables."""
        self.all_shells = []
        self.atom_coords = None
        # self.current_level = None
        self.identifiers_to_shells = {}
        self.level_shells = {}
        self.past_substructs = set()
        self.shells_gen = None

    def initialize_mol(self, mol):
        """Set general properties of `mol` that apply to all its conformers.

        Parameters
        ----------
        mol : RDKit Mol
            Input molecule `Mol` object.
        """
        self.mol = mol
        self.atoms = np.array(
            [x.GetIdx() for x in mol.GetAtoms() if x.GetAtomicNum() > 1]
        )  # ignore hydrogens

        if self.exclude_floating and len(self.atoms) > 1:
            # ignore floating atoms
            self.atoms = np.array(
                [
                    x.GetIdx()
                    for x in mol.GetAtoms()
                    if (x.GetAtomicNum() > 1 and x.GetDegree() > 0)
                ]
            )

        self.bound_atoms_dict = bound_atoms_from_mol(self.mol, self.atoms)
        self.connectivity = {}
        for i, atom1 in enumerate(self.atoms):
            for j, atom2 in enumerate(self.atoms):
                if i <= j:
                    break
                pair = (int(atom1), int(atom2))
                bond = self.mol.GetBondBetweenAtoms(*pair)
                if bond is not None:
                    bond = bond.GetBondType()
                self.connectivity[pair] = self.bond_types[bond]
                self.connectivity[pair[::-1]] = self.connectivity[pair]

        self.initialize_identifiers()

    def initialize_conformer(self, conf):
        """Retrieve atom coordinates and instantiate shells generator.

        Parameters
        ----------
        conf : RDKit Conformer
            Conformer to fingerprint
        """
        self.conf = conf
        self.atom_coords = coords_from_atoms(self.atoms, self.conf)
        self.shells_gen = ShellsGenerator(
            self.conf,
            self.atoms,
            radius_multiplier=self.radius_multiplier,
            atom_coords=self.atom_coords,
            include_disconnected=self.include_disconnected,
            bound_atoms_dict=self.bound_atoms_dict,
        )

    def initialize_identifiers(self):
        """Set initial identifiers for atoms."""
        self.init_identifiers = identifiers_from_invariants(
            self.mol, self.atoms, rdkit_invariants=self.rdkit_invariants
        )

    def __next__(self):
        """Run next iteration of fingerprinting."""
        if self.current_level is None:
            shells_dict = next(self.shells_gen)
            if self.current_level != 0:
                raise Exception(
                    "ShellsGenerator is not at level 0 at start."
                    " This should never happen."
                )

            for atom, shell in shells_dict.items():
                shell.identifier = self.init_identifiers[atom]
                self.identifiers_to_shells.setdefault(
                    shell.identifier, set()
                ).add(shell)
                self.past_substructs.add(shell.substruct)

            level_shells = set(shells_dict.values())
        else:
            # stop if maximum level has been reached
            if self.current_level >= self.level and self.level != -1:
                logging.debug("Hit maximum level")
                raise StopIteration

            # stop if all substructs contain all atoms (there will never
            # be another new substruct), and that substruct has been seen
            if self.remove_duplicate_substructs and all(
                (len(x.substruct.atoms) == len(self.atoms))
                for x in self.shells_gen.get_shells_at_level(
                    self.current_level
                ).values()
            ):
                logging.debug("Ran out of substructs")
                raise StopIteration

            shells_dict = next(self.shells_gen)

            for atom, shell in shells_dict.items():
                identifier = identifier_from_shell(
                    shell,
                    self.atom_coords,
                    self.connectivity,
                    self.current_level,
                    self.stereo,
                )
                shell.identifier = identifier

            accepted_shells = sorted(
                shells_dict.values(), key=self._shell_to_tuple
            )

            # filter shells that correspond to already seen substructs
            if self.remove_duplicate_substructs:
                unique_substruct_shells = []
                for shell in accepted_shells:
                    if shell.substruct in self.past_substructs:
                        logging.debug(
                            (
                                "Shell with identifier {} at level {} is not "
                                "unique. Removing."
                            ).format(shell.identifier, self.current_level)
                        )
                        shell.is_duplicate = True
                        for x in self.past_substructs:
                            if x == shell.substruct:
                                shell.duplicate = x.shell
                                continue
                        continue
                    unique_substruct_shells.append(shell)
                    self.past_substructs.add(shell.substruct)

                accepted_shells = unique_substruct_shells

            # store shells
            for shell in accepted_shells:
                self.identifiers_to_shells.setdefault(
                    shell.identifier, set()
                ).add(shell)

            level_shells = self.level_shells[self.current_level - 1].union(
                set(accepted_shells)
            )
            if len(level_shells) == len(
                self.level_shells[self.current_level - 1]
            ):
                self.shells_gen.back()
                logging.debug("No new shells added. Convergence reached.")
                raise StopIteration

        self.all_shells.extend(shells_dict.values())
        self.level_shells[self.current_level] = level_shells

    next = __next__

    @staticmethod
    def _shell_to_tuple(shell):
        return (shell.identifier, shell.center_atom)

    def get_shells_at_level(self, level=-1, exact=False, atom_mask=set()):
        """Get set of shells at the specified level.

        Parameters
        ----------
        level : int or None, optional
            Level/iteration
        exact : bool, optional
            Exact level
        atom_mask : int or set of int, optional
            Don't return shells whose substructures contain these atoms.

        Returns
        -------
        set of Shell : Shells at level
        """
        if level in (-1, None) or level not in self.level_shells:
            if exact or len(self.level_shells) == 0:
                raise IndexError(
                    "Level {!r} fingerprints have not yet been "
                    "generated".format(level)
                )

            true_level = self.current_level
            if level in (-1, None):
                if self.level not in (-1, None):
                    level = self.level
                else:
                    level = true_level
        else:
            true_level = level

        shells = self.level_shells[true_level]
        if len(atom_mask) > 0:
            try:
                atom_mask = set(atom_mask)
            except TypeError:
                atom_mask = {atom_mask}
            shells = {
                x for x in shells if x.substruct.atoms.isdisjoint(atom_mask)
            }

        return shells

    def get_fingerprint_at_level(
        self, level=-1, bits=None, exact=False, atom_mask=set()
    ):
        """Get the fingerprint at the specified level.

        Parameters
        ----------
        level : int or None, optional
            Level/iteration
        bits : int or None, optional
            Return fingerprints folded to this number of bits. If unspecified,
            defaults to bits set when instantiated.
        exact : bool, optional
            Exact level
        atom_mask : int or set of int, optional
            Don't return shells whose substructures contain these atoms.

        Returns
        -------
        Fingerprint : Fingerprint at level
        """
        if bits in (-1, None):
            bits = self.bits

        shells = self.get_shells_at_level(
            level=level, exact=exact, atom_mask=atom_mask
        )

        identifiers = signed_to_unsigned_int(
            np.array([x.identifier for x in shells], dtype=IDENT_DTYPE)
        )

        fprint = self.fp_type.from_indices(identifiers, level=level)

        return fprint.fold(bits)

    def substructs_to_pdb(
        self,
        level=None,
        bits=None,
        out_dir="substructs",
        reorient=True,
        exact=False,
    ):
        """Save all accepted substructs from current level to PDB.

        Parameters
        ----------
        level : int or None, optional
            Level of fingerprinting/number of iterations
        bits : int or None, optional
            Folding level of identifiers
        out_dir : str, optional
            Directory to which to save PDB files.
        reorient : bool, optional
            Reorient substructure to match stereo quadrants.
        """
        shells = self.get_shells_at_level(level=level, exact=exact)

        if bits in (-1, None):
            bits = self.bits

        touch_dir(out_dir)

        out_files = []
        for shell in shells:
            identifier = signed_to_unsigned_int(shell.identifier) % bits
            out_file = os.path.join(out_dir, "{}.pdb.gz".format(identifier))
            shell_to_pdb(
                self.mol,
                shell,
                self.atom_coords,
                self.bound_atoms_dict,
                out_file,
                reorient=reorient,
            )
            out_files.append(out_file)
        return out_files

    @property
    def current_level(self):
        try:
            return self.shells_gen.level
        except AttributeError:
            return None

    def __iter__(self):
        return self


class ShellsGenerator(object):
    """Generate nested `Shell` objects from molecule upon request."""

    def __init__(
        self,
        conf,
        atoms,
        radius_multiplier=0.5,
        include_disconnected=True,
        atom_coords=None,
        bound_atoms_dict=None,
    ):
        """Initialize the generator.

        After initialization, the generator can be iterated to generate a
        `dict` matching atom ids to that atom's shell at that
        iteration/level.

        Parameters
        ----------
        conf : RDKit Conformer
            Conformer for which to generate shells.
        atoms : list of int
            Atom ids.
        radius_multiplier : float, optional
            Multiplier by which to increase shell radii.
        include_disconnected : bool, optional
            Include atoms not bonded to center atoms within shell.
        atom_coords : dict or None, optional
            Dict matching atom ids to 1-D array of coordinates. If None, will
            be generated.
        bound_atoms_dict : dict or None, optional
            Dict matching atom ids to set of bound atom ids. If None, will be
            generated.
        """
        self.atoms = tuple(atoms)
        self.radius_multiplier = radius_multiplier
        self.shells_dict = {}
        self.include_disconnected = include_disconnected
        self.level = None

        if atom_coords is None:
            atom_coords = coords_from_atoms(self.atoms, conf)
        atom_coords = [atom_coords.get(x) for x in self.atoms]
        self.distance_matrix = array_ops.make_distance_matrix(atom_coords)

        overlap_atoms = [
            (self.atoms[i], self.atoms[j])
            for i, j in zip(*np.where(self.distance_matrix <= array_ops.EPS))
            if i < j
        ]
        if len(overlap_atoms) > 0:
            logging.warning(
                "Overlapping atoms {} in conformer {} of molecule"
                " {}. Fingerprinting will continue but is less "
                "reliable.".format(
                    ", ".join(map(repr, overlap_atoms)),
                    conf.GetId(),
                    conf.GetOwningMol().GetProp("_Name"),
                )
            )

        if not include_disconnected and bound_atoms_dict is None:
            bound_atoms_dict = bound_atoms_from_mol(conf.GetOwningMol(), atoms)
        self.bound_atoms_dict = bound_atoms_dict

    def get_match_atoms(self, rad):
        """Get atoms within shell at radius `rad`.

        Parameters
        ----------
        rad : float
            Radius of shell.

        Returns
        -------
        dict : Dict matching atom id to set of ids for other atoms within
               shell
        """
        match_atoms_dict = {x: set() for x in self.atoms}
        atom_pair_indices_list = list(
            zip(*np.where(self.distance_matrix <= rad))
        )
        for i, j in atom_pair_indices_list:
            if i <= j:
                continue
            atom1, atom2 = (self.atoms[i], self.atoms[j])
            match_atoms_dict[atom1].add(atom2)
            match_atoms_dict[atom2].add(atom1)
        if not self.include_disconnected:
            match_atoms_dict = {
                k: v.intersection(self.bound_atoms_dict[k])
                for k, v in match_atoms_dict.items()
            }
        return match_atoms_dict

    def __next__(self):
        """Get next iteration's `dict` of atom shells."""
        if self.level is None:
            self.level = 0
            self.shells_dict[self.level] = {
                x: Shell(x, radius=0.0) for x in self.atoms
            }
            return self.shells_dict[self.level]

        self.level += 1
        self.shells_dict[self.level] = {}
        rad = self.level * self.radius_multiplier
        match_atoms_dict = self.get_match_atoms(rad)
        for atom in self.atoms:
            match_atoms = match_atoms_dict[atom]
            last_match_shells = [
                self.shells_dict[self.level - 1].get(x) for x in match_atoms
            ]
            last_shell = self.shells_dict[self.level - 1][atom]
            shell = Shell(
                atom, last_match_shells, radius=rad, last_shell=last_shell
            )
            self.shells_dict[self.level][atom] = shell
        return self.shells_dict[self.level]

    next = __next__

    def back(self):
        """Back up one iteration."""
        if self.level in {None, 0}:
            return
        del self.shells_dict[self.level]
        self.level -= 1

    def get_shells_at_level(self, level):
        """Get `dict` of atom shells at specified level/iteration.

        If not run to `level`, raises `IndexError`.

        Parameters
        ----------
        level : int
            Level/iteration from which to retrieve shells `dict`.

        Returns
        -------
        dict: Dict matching atom ids to that atom's `Shell` at that level.
        """
        if level not in self.shells_dict:
            raise IndexError(
                "Level {!r} shells have not been generated".format(level)
            )
        return self.shells_dict[level]

    def __iter__(self):
        return self


# Getting atom properties
def coords_from_atoms(atoms, conf):
    """Build `dict` matching atom id to coordinates.

    Parameters
    ----------
    atoms : list of int
        Atom ids
    conf : RDKit Conformer
        Conformer from which to fetch coordinates

    Returns
    -------
    dict : Dict matching atom id to 1-D array of coordinates.
    """
    coordinates = [
        np.array(conf.GetAtomPosition(int(x)), dtype=np.float64) for x in atoms
    ]
    return dict(zip(atoms, coordinates))


def bound_atoms_from_mol(mol, atoms):
    """Build `dict` matching atom id to ids of bounded atoms.

    Bound atoms not in `atoms` are ignored.

    Parameters
    ----------
    mol : RDKit Mol
        Input mol
    atoms : list of int
        List of atom IDs

    Returns
    -------
    dict : Dict matching atom id to set of bound atom ids.
    """
    atoms = set(atoms)
    bound_atoms_dict = {}
    for bond in mol.GetBonds():
        atom1, atom2 = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        if (atom1 not in atoms) or (atom2 not in atoms):
            continue
        bound_atoms_dict.setdefault(atom1, set()).add(atom2)
        bound_atoms_dict.setdefault(atom2, set()).add(atom1)
    return bound_atoms_dict


# Generating identifiers
def hash_int64_array(array, seed=MMH3_SEED):
    """Hash an int64 array into a 32-bit integer.

    Parameters
    ----------
    array : ndarray of int64
        Numpy array containing integers
    seed : any, optional
        Seed for MurmurHash3.

    Returns
    -------
    int : 32-bit integer
    """
    if not np.issubdtype(array.dtype, IDENT_DTYPE):
        raise TypeError(
            "Provided array has dtype {} not {}".format(
                array.dtype, IDENT_DTYPE.__name__
            )
        )
    # ensure all hashed integers are positive
    hashed_int = mmh3.hash(array, seed)
    return hashed_int


def signed_to_unsigned_int(a, bits=BITS):
    """Convert `int` between +/-`bits` to an int between 0 and `bits`.

    Parameters
    ----------
    a : int or ndarray of int
        Integer
    bits : int, optional
        Maximum size of int. E.g. 32-bit is 2^32.

    Returns
    -------
    int : unsigned integer
    """
    return (a + bits) % bits


def identifiers_from_invariants(
    mol, atoms, rdkit_invariants=RDKIT_INVARIANTS_DEF
):
    """Initialize ids according to Daylight invariants.

    Parameters
    ----------
    mol : RDKit Mol
        Input molecule
    atoms : list of int
        IDs for atoms in mol for which to generate identifiers.
    rdkit_invariants : bool, optional
        Use the atom invariants used by RDKit for its Morgan fingerprint.

    Returns
    -------
    ndarray of int64 : initial identifiers for atoms
    """
    if rdkit_invariants:
        identifiers = [
            hash_int64_array(
                rdkit_invariants_from_atom(mol.GetAtomWithIdx(int(x)))
            )
            for x in atoms
        ]
    else:
        identifiers = [
            hash_int64_array(invariants_from_atom(mol.GetAtomWithIdx(int(x))))
            for x in atoms
        ]
    atom_to_identifier_dict = dict(zip(atoms, identifiers))
    return atom_to_identifier_dict


def invariants_from_atom(atom):
    """Get seven invariants from atom.

    Invariants used are the six Daylight invariants, plus an indicator of
    whether the atom is in a ring, as detailed in [1].

    References
    ----------
    1. D Rogers, M Hahn. J. Chem. Inf. Model., 2010, 50 (5), pp 742-754
       https://doi.org/10.1021/ci100050t

    Parameters
    ----------
    atom : RDKit Atom
        Input atom

    Returns
    -------
    1-D array if int64: Array of 7 invariants
    """
    num_hs = atom.GetTotalNumHs()
    return np.array(
        [
            atom.GetTotalDegree() - num_hs,  # Num heavy neighbors
            atom.GetTotalValence() - num_hs,
            atom.GetAtomicNum(),
            int(atom.GetMass()),
            atom.GetFormalCharge(),
            num_hs,
            int(atom.IsInRing()),
        ],
        dtype=IDENT_DTYPE,
    )


def rdkit_invariants_from_atom(atom):
    """Get the 6 atom invariants RDKit uses for its Morgan fingerprints.

    Parameters
    ----------
    atom : RDKit Atom
        Input atom

    Returns
    -------
    1-D array if int64: Array of 6 invariants
    """
    delta_mass = int(
        atom.GetMass()
        - Chem.GetPeriodicTable().GetAtomicWeight(atom.GetAtomicNum())
    )
    return np.array(
        [
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            atom.GetTotalNumHs(),
            atom.GetFormalCharge(),
            delta_mass,
            int(atom.IsInRing()),
        ],
        dtype=IDENT_DTYPE,
    )


def identifier_from_shell(shell, atom_coords, connectivity, level, stereo):
    """Determine new identifier for a shell at a specific level.

    Parameters
    ----------
    shell : Shell
        Shell for which to determine identifier
    atom_coords : dict
        Dict matching atom ids to coords.
    connectivity : dict
        Dict matching atom id pair tuples to their bond order (5 for unbound).
    level : int
        Level/iteration
    stereo : bool
        Add stereo indicators
    """
    header = [level, shell.last_shell.identifier]
    atom_tuples = atom_tuples_from_shell(
        shell, atom_coords, connectivity, stereo
    )
    flat_atom_tuples = [y for x in atom_tuples for y in x]
    arr = np.array(header + flat_atom_tuples, dtype=IDENT_DTYPE)
    return hash_int64_array(arr)


def atom_tuples_from_shell(shell, atom_coords, connectivity, stereo):
    """Generate sorted atom tuples for neighboring atoms.

    Parameters
    ----------
    shell : Shell
        Shell for which to build atom tuples
    atom_coords : dict
        Dict matching atom ids to coords.
    connectivity : dict
        Dict matching atom id pair tuples to their bond order (5 for unbound).
    stereo : bool
        Add stereo indicators to tuples
    """
    if len(shell.shells) == 0:
        return []

    # create list matching "bond order" to last iteration's atom identifier
    atom_tuples = [
        (connectivity[(shell.center_atom, x.center_atom)], x.identifier, x)
        for x in shell.shells
    ]

    # add stereo indicator
    if stereo:
        atom_tuples.sort(key=_first_two)
        stereo_indicators = stereo_indicators_from_shell(
            shell, atom_tuples, atom_coords
        )
        atom_tuples = [
            x[:-1] + (y,) + (x[-1],)
            for x, y in zip(atom_tuples, stereo_indicators)
        ]

    # final sort
    atom_tuples = [x[:-1] for x in atom_tuples]
    atom_tuples.sort()
    return atom_tuples


def _first_two(xs):
    return (xs[0], xs[1])


# Methods used for stereo indicators
def pick_y(atom_tuples, cent_coords, y_precision=Y_AXIS_PRECISION):
    """Pick a y-coordinate from atom tuples or mean coordinate.

    Parameters
    ----------
    atom_tuples : list of tuple
        Sorted list of atom tuples
    cent_coords : Nx3 array of float
        Coordinates of atoms with center atom at origin.
    y_precision : str, optional
        For mean to be chosen for y-coordinate, it must be at least this
        distance from the origin. Useful when atoms are symmetrical around
        the center atom where a slight shift in any atom results in a very
        different y.

    Returns
    -------
    1x3 array of float or None : y-coordinate
    int or None : index to y-atom, if y was chosen from the atoms.
    """
    y_inds = get_first_unique_tuple_inds(atom_tuples, 1, assume_sorted=True)
    # select y-axis vector
    if len(y_inds) > 0:  # unique atom could be found
        y = cent_coords[y_inds, :]
        return y, y_inds[0]
    elif len(atom_tuples) == 2:  # exactly two identical atoms
        y = cent_coords[0, :]
        return y, 0
    else:  # no unique atom could be found
        # assign y-axis as vector to mean coordinate
        y = np.mean(cent_coords, axis=0)
        if np.linalg.norm(y) < y_precision:
            return None, None
        else:
            return y, None


def pick_z(
    connectivity,
    identifiers,
    cent_coords,
    y,
    long_angle,
    z_precision=Z_AXIS_PRECISION,
):
    """Pick a z-coordinate orthogonal to `y`.

    Parameters
    ----------
    connectivity : dict
        Dict matching atom id pair tuples to their bond order (5 for unbound).
    identifiers : iterable of int
        Atom identifiers
    cent_coords : Nx3 array of float
        Coordinates of atoms with center atom at origin.
    y : 1x3 array of float
        y-coordinate
    long_angle : Nx1 array of float
        Absolute angle of atoms from orthogonal to `y`.
    z_precision : str, optional
        Minimum difference in `long_angle` between two potential z-atoms.
        Used as a tie breaker to prevent small shift in one atom resulting
        in very different z.

    Returns
    -------
    1x3 array of float or None : z-coordinate
    """
    angle_from_right = sorted(
        zip(
            np.asarray(long_angle / z_precision, dtype=np.int),
            connectivity,
            identifiers,
            range(len(identifiers)),
        )
    )

    z_angle_inds = get_first_unique_tuple_inds(
        angle_from_right, 1, assume_sorted=True
    )

    if len(z_angle_inds) > 0:
        z_ind = angle_from_right[z_angle_inds[0]][-1]
        z = array_ops.project_to_plane(cent_coords[z_ind, :], y)
        return z
    else:
        return None


def stereo_indicators_from_shell(
    shell, atom_tuples, atom_coords_dict, add_transform_to_shell=True
):
    """Get `list` of `int` indicating location of atoms on unit sphere.

    Parameters
    ----------
    shell : Shell
        Shell for which to get stereo indicators.
    atom_tuples : list of tuple
        List of atom tuples.
    atom_coords_dict : dict
        Dict matching atom ids to coords.
    add_transform_to_shell : bool, optional
        Calculate transformation matrix to align coordinates to unit sphere,
        and add to shell.

    Returns
    -------
    list of int : stereo indicators for atoms in `atom_tuples`.
    """
    cent_coord = atom_coords_dict[shell.center_atom]

    y = None
    z = None
    if len(atom_tuples) > 0:
        (connectivity, identifiers, shells) = list(zip(*atom_tuples))

        stereo_indicators = np.zeros((len(atom_tuples),), dtype=IDENT_DTYPE)
        atoms = [x.center_atom for x in shells]
        mask = np.ones(len(atom_tuples), dtype=np.bool)

        cent_coords = (
            np.array(
                [atom_coords_dict.get(x) for x in atoms], dtype=np.float64
            )
            - cent_coord
        )
        # mask atom lying on center atom from consideration
        cent_overlap_indices = np.all(cent_coords == np.zeros(3), axis=1)
        mask[cent_overlap_indices] = False

        # pick y based on first unique atom tuple or mean
        y, y_ind = pick_y(atom_tuples, cent_coords)
        if y_ind is not None:
            mask[y_ind] = False  # use to ensure y not chosen for z

        if y is not None:  # y was picked
            # pick z based on closeness to pi/2 from y-axis
            long_angle = np.pi / 2.0 - array_ops.calculate_angles(
                cent_coords, y
            )
            # perfect right angles
            long_angle[np.fabs(long_angle) < array_ops.EPS] = 0.0
            long_sign = np.asarray(np.sign(long_angle), dtype=IDENT_DTYPE)
            long_sign[long_sign == 0] = 1
            long_angle = np.fabs(long_angle)
            tmp_conn = np.array(connectivity, dtype=IDENT_DTYPE)[mask]
            tmp_ident = np.array(identifiers, dtype=IDENT_DTYPE)[mask]
            z = pick_z(
                tmp_conn, tmp_ident, cent_coords[mask], y, long_angle[mask]
            )

            if z is not None:  # z was picked
                quad_indicators = quad_indicators_from_coords(
                    cent_coords, y, y_ind, z, long_sign
                )
                stereo_indicators = quad_indicators

            #  set indicators for atoms near poles to +/-1
            pole_inds = np.pi / 2 - long_angle < POLAR_CONE_RAD
            stereo_indicators[pole_inds] = long_sign[pole_inds]

        stereo_indicators[cent_overlap_indices] = 0
        stereo_indicators = stereo_indicators.flatten().tolist()
    else:
        stereo_indicators = []

    if add_transform_to_shell:
        shell.transform_matrix = array_ops.make_transform_matrix(
            cent_coord, y, z
        )

    return stereo_indicators


def quad_indicators_from_coords(cent_coords, y, y_ind, z, long_sign):
    """Create angle indicators for four quadrants in each hemisphere.

    Parameters
    ----------
    cent_coords : Nx3 array of float
        Array of centered coordinates.
    y : 1-D array of float
        Vector lying along y-axis.
    y_ind : int
        Index of `cent_coords` corresponding to `y`.
    z : 1-D array of float
        Vector lying along z-axis
    long_sign : Nx1 array of int
        Array of signs of vectors in `cent_coords` indicating whether they are
        above (+1) or below (-1) the xz-plane.

    Returns
    -------
    Nx1 array of int : Quadrant indicators. Clockwise from `z` around `y`,
                    indicators are 2, 3, 4, 5 for vectors above the xz-plane
                    and -2, -3, -4, -5 for vectors below the xz-plane.
    """
    atom_lats = array_ops.project_to_plane(cent_coords, y)

    with np.errstate(invalid="ignore"):  # y atom_lat should be (0, 0, 0)
        angle_from_z = array_ops.calculate_angles(atom_lats, z, y).flatten()

    if y_ind is not None:
        angle_from_z[y_ind] = 0.0  # otherwise, will be nan

    # offset by pi/4 so z-axis isn't an edge case
    lat_angle = array_ops.rotate_angles(angle_from_z, np.pi / 4)

    # create quadrant indicators
    quad_indicators = 2 + np.asarray(
        lat_angle * 4 / (2 * np.pi), dtype=IDENT_DTYPE
    )
    quad_indicators *= long_sign
    return quad_indicators


def get_first_unique_tuple_inds(
    tuples_list, num_ret, ignore=[], assume_sorted=True
):
    """Return indices of first `num_ret` unique tuples in a list.

    Only first 2 values of each tuple are considered.

    Parameters
    ----------
    tuples_list : list of tuple
        List of tuples. Only first two unique values are considered.
    num_ret : int
        Maximum number of first unique tuples to return.
    ignore : list, optional
        Indices for tuples not be considered as unique.
    assume_sorted : bool, optional
        If True, assume list is already sorted by tuples.

    Returns
    -------
    tuple of int : List of at most `num_ret` ints indicating index of
                   unique tuples in list.
    """
    unique = {}
    repeated = set(tuples_list[x][:2] for x in ignore)
    for i, tup in enumerate(tuples_list):
        val = tup[:2]
        if val in repeated:
            if assume_sorted and len(unique) == num_ret:
                break  # okay to stop looking
        elif val in unique:
            del unique[val]
            repeated.add(val)
        else:
            unique[val] = i

    return tuple(sorted(unique.values())[:num_ret])

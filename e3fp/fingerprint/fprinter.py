"""Tools for generating E3FP fingerprints.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
from __future__ import division, print_function
import os
import logging

import numpy as np
from scipy.spatial.distance import pdist, squareform

from rdkit import Chem
import mmh3

from python_utilities.io_tools import smart_open, touch_dir
from python_utilities.scripting import setup_logging
from e3fp.fingerprint.structs import Shell
from e3fp.fingerprint.fprint import Fingerprint, CountFingerprint

IDENT_DTYPE = np.int64  # np.dtype to use for identifiers
X_AXIS, Y_AXIS, Z_AXIS = np.identity(3, dtype=np.float64)
Y_AXIS_PRECISION = 0.1  # angstroms
Z_AXIS_PRECISION = 0.01  # rad
EPS = 1e-12  # epsilon, a number close to 0
POLAR_CONE_RAD = np.pi / 36  # rad
MMH3_SEED = 0
BOND_TYPES = {None: 5, Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2,
              Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}
PDB_LINE = ("HETATM{atom_id:>5d} {name:<4s} LIG A   1    "
            "{coord[0]:>8.3f}{coord[1]:>8.3f}{coord[2]:>8.3f}"
            "{occupancy:>6.2f}{temp:>6.2f}          {elem:>2s}{charge:>2s}")
BITS = 2**32

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
        Instead of simple bit-based ``Fingerprint`` object, generate
        ``CountFingerprint`` that tracks number of times each bit appears in a
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

    Attributes
    ----------
    current_level : int
        The maximum level/iteration to which the fingerprinter has been run
        on the current conformer.
    level_shells : dict
        Dict matching level to set of all shells accepted at that level.
    """

    def __init__(self, bits=BITS, level=-1, radius_multiplier=2.0,
                 stereo=False, counts=False, remove_duplicate_substructs=True,
                 include_disconnected=True):
        """Initialize fingerprinter settings."""
        self.mol = None
        if level is None:
            level = -1
        self.level = level
        if not np.log2(bits).is_integer():
            logging.warning("bits are not multiple of 2. Multiples of 2 are "
                            "highly recommended")
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
                """No termination condition specified. 'level' must be
                provided or 'remove_duplicate_substructs' must be True""")

        self.include_disconnected = include_disconnected

        self.bond_types = BOND_TYPES
        self.reset()

    def run(self, conf=None, mol=None, conf_id=None, return_substruct=False):
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
            except AttributeError:  # conf is conf id; use existing mol
                mol = self.mol
        else:
            try:
                conf.GetOwningMol()
            except AttributeError:  # conf isn't RDKit Conformer
                try:
                    conf = mol.GetConformers()[conf]
                except TypeError:  # conf isn't ID either. Fall back to first
                    conf = mol.GetConformers()[0]

        if mol is not self.mol:
            self.reset_mol()
            self.initialize_mol(mol)

        self.initialize_conformer(conf)

        for i in iter(self):
            pass

    def reset(self):
        """Clear all variables associated with the last run."""
        self.reset_mol()

    def reset_conf(self):
        """Clear only conformer-specific variables."""
        self.all_shells = []
        self.atom_coords = None
        self.current_level = None
        self.identifiers_to_shells = {}
        self.level_shells = {}
        self.past_substructs = set()
        self.shells_gen = None

    def reset_mol(self):
        """Clear all variables associated with the molecule."""
        self.atoms = None
        self.bound_atoms_dict = {}
        self.connectivity = {}
        self.init_identifiers = {}
        self.reset_conf()

    def initialize_mol(self, mol):
        """Set general properties of `mol` that apply to all its conformers.

        Parameters
        ----------
        mol : RDKit Mol
            Input molecule ``Mol`` object.
        """
        self.mol = mol
        self.atoms = np.array([x.GetIdx() for x in mol.GetAtoms()
                               if x.GetAtomicNum() > 1])  # ignore hydrogens
        self.bound_atoms_dict = bound_atoms_from_mol(self.mol, self.atoms)
        self.connectivity = {}
        for i, atom1 in enumerate(self.atoms):
            for j, atom2 in enumerate(self.atoms):
                if i <= j:
                    break
                pair = (atom1, atom2)
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
            self.conf, self.atoms, radius_multiplier=self.radius_multiplier,
            atom_coords=self.atom_coords,
            include_disconnected=self.include_disconnected,
            bound_atoms_dict=self.bound_atoms_dict)

    def initialize_identifiers(self):
        """Set initial identifiers for atoms."""
        self.init_identifiers = identifiers_from_invariants(self.mol,
                                                            self.atoms)

    def next(self):
        """Run next iteration of fingerprinting."""
        if self.current_level is None:
            shells_dict = self.shells_gen.next()
            self.current_level = 0
            if self.shells_gen.level != 0:
                raise Exception("ShellGenerator is not at level 0 at start."
                                " This should never happen.")

            for atom, shell in shells_dict.items():
                shell.identifier = self.init_identifiers[atom]
                self.identifiers_to_shells.setdefault(shell.identifier,
                                                      set()).add(shell)

            shells = shells_dict.values()
            self.all_shells.extend(shells)
            self.level_shells[0] = set(shells)

            self.past_substructs.update([x.substruct for x in shells])
        else:
            # stop if maximum level has been reached
            if (self.current_level >= self.level and
                    self.level not in (-1, None)):
                raise StopIteration

            # stop if all substructs contain all atoms (there will never
            # be another new substruct), and that substruct has been seen
            if (self.remove_duplicate_substructs and
                all((len(x.substruct.atoms) == len(self.atoms))
                    for x in self.shells_gen.get_shells_at_level(
                        self.current_level).itervalues())):

                raise StopIteration

            shells_dict = next(self.shells_gen)
            self.current_level = self.shells_gen.level

            for atom, shell in shells_dict.iteritems():
                identifier = identifier_from_shell(shell, self.atom_coords,
                                                   self.connectivity,
                                                   self.current_level,
                                                   self.stereo)
                shell.identifier = identifier

            accepted_shells = sorted(shells_dict.values(),
                                     key=self._shell_to_tuple)

            # filter shells that correspond to already seen substructs
            if self.remove_duplicate_substructs:
                unique_substruct_shells = []
                for shell in accepted_shells:
                    if shell.substruct in self.past_substructs:
                        continue
                    unique_substruct_shells.append(shell)
                    self.past_substructs.add(shell.substruct)

                accepted_shells = unique_substruct_shells

            # store shells
            for shell in accepted_shells:
                self.identifiers_to_shells.setdefault(shell.identifier,
                                                      set()).add(shell)

            level_shells = self.level_shells[self.current_level - 1].union(
                set(accepted_shells))
            if (len(level_shells) == len(
                    self.level_shells[self.current_level - 1])):
                self.current_level -= 1
                raise StopIteration

            self.all_shells.extend(shells_dict.values())
            self.level_shells[self.current_level] = level_shells

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
                raise IndexError("Level {!r} fingerprints have not yet been "
                                 "generated".format(level))

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
            shells = {x for x in shells
                      if x.substruct.atoms.isdisjoint(atom_mask)}

        return shells

    def get_fingerprint_at_level(self, level=-1, bits=None, exact=False,
                                 atom_mask=set()):
        """Summary

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

        shells = self.get_shells_at_level(level=level, exact=exact,
                                          atom_mask=atom_mask)

        identifiers = signed_to_unsigned_int(
            np.array([x.identifier for x in shells], dtype=IDENT_DTYPE))

        fprint = self.fp_type.from_indices(identifiers, level=level)

        return fprint.fold(bits)

    def substructs_to_pdb(self, bits=None, out_dir='substructs',
                          reorient=True):
        """Save all accepted substructs from current level to PDB.

        Parameters
        ----------
        bits : int or None, optional
            Folding level of identifiers
        out_dir : str, optional
            Directory to which to save PDB files.
        reorient : bool, optional
            Reorient substructure to match stereo quadrants.
        """
        if bits in (-1, None):
            bits = self.bits
        touch_dir(out_dir)

        for shell in self.level_shells[self.current_level]:
            identifier = signed_to_unsigned_int(shell.identifier) % bits
            out_file = os.path.join(out_dir, "{}.pdb.gz".format(identifier))
            shell_to_pdb(self.mol, shell, self.atom_coords,
                         self.bound_atoms_dict, out_file, reorient=reorient)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()


class ShellsGenerator(object):

    """Generate nested ``Shell`` objects from molecule upon request."""

    def __init__(self, conf, atoms, radius_multiplier=0.5,
                 include_disconnected=True, atom_coords=None,
                 bound_atoms_dict=None):
        """Initialize the generator.

        After initialization, the generator can be iterated to generate a
        ``dict`` matching atom ids to that atom's shell at that
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
        atom_coords = map(atom_coords.get, self.atoms)
        self.distance_matrix = build_distance_matrix(atom_coords)

        if not include_disconnected and bound_atoms_dict is None:
            bound_atoms_dict = bound_atoms_from_mol(conf.GetOwningMol(),
                                                    atoms)
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
        atom_pair_indices_list = zip(*np.where(self.distance_matrix <= rad))
        for i, j in atom_pair_indices_list:
            if i <= j:
                continue
            atom1, atom2 = (self.atoms[i], self.atoms[j])
            match_atoms_dict[atom1].add(atom2)
            match_atoms_dict[atom2].add(atom1)
        if not self.include_disconnected:
            match_atoms_dict = {k: v.intersection(self.bound_atoms_dict[k])
                                for k, v in match_atoms_dict.iteritems()}
        return match_atoms_dict

    def next(self):
        """Get next iteration's ``dict`` of atom shells."""
        if self.level is None:
            self.level = 0
            self.shells_dict[self.level] = {x: Shell(x, radius=0.)
                                            for x in self.atoms}
            return self.shells_dict[self.level]

        self.level += 1
        self.shells_dict[self.level] = {}
        rad = self.level * self.radius_multiplier
        match_atoms_dict = self.get_match_atoms(rad)
        for atom in self.atoms:
            match_atoms = match_atoms_dict[atom]
            last_match_shells = map(self.shells_dict[self.level - 1].get,
                                    match_atoms)
            last_shell = self.shells_dict[self.level - 1][atom]
            shell = Shell(atom, last_match_shells, radius=rad,
                          last_shell=last_shell)
            self.shells_dict[self.level][atom] = shell
        return self.shells_dict[self.level]

    def get_shells_at_level(self, level):
        """Get ``dict`` of atom shells at specified level/iteration.

        If not run to `level`, raises ``IndexError``.

        Parameters
        ----------
        level : int
            Level/iteration from which to retrieve shells ``dict``.

        Returns
        -------
        dict: Dict matching atom ids to that atom's ``Shell`` at that level.
        """
        if level not in self.shells_dict:
            raise IndexError(
                "Level {r!} shells have not been generated".format(level))
        return self.shells_dict[level]

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()


# Getting atom properties
def coords_from_atoms(atoms, conf):
    """Build ``dict`` matching atom id to coordinates.

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
    coordinates = [np.array(x, dtype=np.float64) for x
                   in map(conf.GetAtomPosition, atoms)]
    return dict(zip(atoms, coordinates))


def bound_atoms_from_mol(mol, atoms):
    """Build ``dict`` matching atom id to ids of bounded atoms.

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
            "Provided array has dtype {} not {}".format(array.dtype,
                                                        IDENT_DTYPE.__name__))
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


def identifiers_from_invariants(mol, atoms):
    """Initialize ids according to Daylight invariants.

    Parameters
    ----------
    mol : RDKit Mol
        Input molecule
    atoms : list of int
        IDs for atoms in mol for which to generate identifiers.

    Returns
    -------
    ndarray of int64 : initial identifiers for atoms
    """
    identifiers = map(hash_int64_array,
                      map(invariants_from_atom,
                          map(mol.GetAtomWithIdx, atoms)))
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
    return np.array([atom.GetTotalDegree() - num_hs, atom.GetTotalValence() -
                     num_hs, atom.GetAtomicNum(), int(atom.GetMass()),
                     atom.GetFormalCharge(), num_hs, int(atom.IsInRing())],
                    dtype=IDENT_DTYPE)


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
    atom_tuples = atom_tuples_from_shell(shell, atom_coords, connectivity,
                                         stereo)
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
        for x in shell.shells]
    atom_tuples.sort()

    # add stereo indicator
    if stereo:
        stereo_indicators = stereo_indicators_from_shell(shell, atom_tuples,
                                                         atom_coords)
        atom_tuples = [x[:-1] + (y,) + (x[-1],) for x, y
                       in zip(atom_tuples, stereo_indicators)]
        atom_tuples.sort()

    # final sort
    atom_tuples = [x[:-1] for x in atom_tuples]
    atom_tuples.sort()
    return atom_tuples


# Methods used for stereo indicators
def pick_y(atom_tuples, cent_coords, y_precision=Y_AXIS_PRECISION):
    """Pick a y-coordinate from atom tuples or mean coordinate.

    Parameters
    ----------
    atom_tuples : list of tuple
        Sorted list of atom tuples
    cent_coords : Nx3 float array
        Coordinates of atoms with center atom at origin.
    y_precision : str, optional
        For mean to be chosen for y-coordinate, it must be at least this
        distance from the origin. Useful when atoms are symmetrical around
        the center atom where a slight shift in any atom results in a very
        different y.

    Returns
    -------
    1x3 float array or None : y-coordinate
    int or None : index to y-atom, if y was chosen from the atoms.
    """
    y_inds = get_first_unique_tuple_inds(atom_tuples, 1,
                                         assume_sorted=True)
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


def pick_z(connectivity, identifiers, cent_coords, y, long_angle,
           z_precision=Z_AXIS_PRECISION):
    """Pick a z-coordinate orthogonal to `y`.

    Parameters
    ----------
    connectivity : dict
        Dict matching atom id pair tuples to their bond order (5 for unbound).
    identifiers : iterable of int
        Atom identifiers
    cent_coords : Nx3 float array
        Coordinates of atoms with center atom at origin.
    y : 1x3 float array
        y-coordinate
    long_angle : Nx1 float array
        Absolute angle of atoms from orthogonal to `y`.
    z_precision : str, optional
        Minimum difference in `long_angle` between two potential z-atoms.
        Used as a tie breaker to prevent small shift in one atom resulting
        in very different z.

    Returns
    -------
    1x3 float array or None : z-coordinate
    """
    angle_from_right = sorted(
        zip(np.asarray(long_angle / z_precision, dtype=np.int),
            connectivity,
            identifiers,
            range(len(identifiers))))

    z_angle_inds = get_first_unique_tuple_inds(angle_from_right, 1,
                                               assume_sorted=True)

    if len(z_angle_inds) > 0:
        z_ind = angle_from_right[z_angle_inds[0]][-1]
        z = project_to_plane(cent_coords[z_ind, :], y)
        return z
    else:
        return None


def stereo_indicators_from_shell(shell, atom_tuples, atom_coords_dict,
                                 add_transform_to_shell=True):
    """Get ``list`` of ``int``s indicating location of atoms on unit sphere.

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
        (connectivity, identifiers, shells) = zip(*atom_tuples)

        stereo_indicators = np.zeros((len(atom_tuples),), dtype=IDENT_DTYPE)
        atoms = [x.center_atom for x in shells]
        mask = np.ones(len(atom_tuples), dtype=np.bool)

        cent_coords = np.array(map(atom_coords_dict.get, atoms),
                               dtype=np.float64) - cent_coord

        # pick y based on first unique atom tuple or mean
        y, y_ind = pick_y(atom_tuples, cent_coords)
        if y_ind is not None:
            mask[y_ind] = False  # use to ensure y not chosen for z

        if y is not None:  # y was picked
            # pick z based on closeness to pi/2 from y-axis
            long_angle = np.pi/2 - calculate_angles(cent_coords, y)
            long_angle[np.fabs(long_angle) < EPS] = 0.  # perfect right angles
            long_sign = np.asarray(np.sign(long_angle), dtype=IDENT_DTYPE)
            long_sign[np.where(long_sign == 0)] = 1
            long_angle = np.fabs(long_angle)
            tmp_conn = np.array(connectivity, dtype=IDENT_DTYPE)[mask]
            tmp_ident = np.array(identifiers, dtype=IDENT_DTYPE)[mask]
            z = pick_z(tmp_conn, tmp_ident, cent_coords[mask], y,
                       long_angle[mask])

            if z is not None:  # z was picked
                quad_indicators = quad_indicators_from_coords(cent_coords, y,
                                                              y_ind, z,
                                                              long_sign)
                stereo_indicators = quad_indicators

            #  set indicators for atoms near poles to +/-1
            pole_inds = np.where(np.pi/2 - long_angle < POLAR_CONE_RAD)
            stereo_indicators[pole_inds] = long_sign[pole_inds]

        stereo_indicators = stereo_indicators.flatten().tolist()
    else:
        stereo_indicators = []

    if add_transform_to_shell:
        shell.transform_matrix = make_transform_matrix(cent_coord, y, z)

    return stereo_indicators


def quad_indicators_from_coords(cent_coords, y, y_ind, z, long_sign):
    """Create angle indicators for four quadrants in each hemisphere.

    Parameters
    ----------
    cent_coords : Nx3 float array
        Array of centered coordinates.
    y : 1-D float array
        Vector lying along y-axis.
    y_ind : int
        Index of `cent_coords` corresponding to `y`.
    z : 1-D float array
        Vector lying along z-axis
    long_sign : Nx1 int array
        Array of signs of vectors in `cent_coords` indicating whether they are
        above (+1) or below (-1) the xz-plane.

    Returns
    -------
    Nx1 int array : Quadrant indicators. Clockwise from `z` around `y`,
                    indicators are 2, 3, 4, 5 for vectors above the xz-plane
                    and -2, -3, -4, -5 for vectors below the xz-plane.
    """
    atom_lats = project_to_plane(cent_coords, y)

    angle_from_z = calculate_angles(atom_lats, z, y).flatten()
    if y_ind is not None:
        angle_from_z[y_ind] = 0.  # otherwise, will be nan

    # offset by pi/4 so z-axis isn't an edge case
    lat_angle = rotate_angles(angle_from_z, np.pi / 4)

    # create quadrant indicators
    quad_indicators = 2 + np.asarray(lat_angle * 4/(2 * np.pi),
                                     dtype=IDENT_DTYPE)
    quad_indicators *= long_sign
    return quad_indicators


def get_first_unique_tuple_inds(tuples_list, num_ret, ignore=[],
                                assume_sorted=True):
    """Return indices of first `num_ret` unique tuples in a list.

    Only first 2 values of each tuple are considered.

    Parameters
    ----------
    tuples_list : list of tuple
        List of tuples. Only first two unique values are considered.
    num_ret : int
        Maximum number of first unique tuples to return.
    ignore : list, optional (default [])
        Indices for tuples not be considered as unique.
    assume_sorted : bool, optional (default True)
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


# Vector Algebra Methods
def get_length_vector(v, n=1., axis=1):
    """Return array of length `n` vectors parallel to vectors in `v`.

    Parameters
    ----------
    v : ndarray of float
    n : float, optional
        Length to which to normalize vector.
    axis : int, optional
        Axis along which to normalize length.

    Returns
    -------
    ndarray of float : Axis 1 has length `n`
    """
    u = np.array(v, dtype=np.float64, copy=True)
    if u.ndim == 1:
        mag = np.sqrt(np.dot(u, u))
    else:
        mag = np.atleast_1d(np.sum(u*u, axis))
        np.sqrt(mag, mag)
        if axis is not None:
            mag = np.expand_dims(mag, axis)
    u /= mag
    return u


def get_unit_vector(v, axis=1):
    """Return array of unit vectors parallel to vectors in `v`.

    Parameters
    ----------
    v : ndarray of float
    axis : int, optional
        Axis along which to normalize length.

    Returns
    -------
    ndarray of float : Unit vector of `v`, i.e. `v` divided by its
                       magnitude along `axis`.
    """
    return get_length_vector(v, n=1., axis=axis)


def build_distance_matrix(atom_coords):
    """Build pairwise distance matrix from conformer coordinates.

    Parameters
    ----------
    atom_coords : ndarray of float
        an Mx3 array of xyz atom coordinates.

    Returns
    -------
    ndarray of float : square symmetrical distance matrix with indices
                       corresponding to indices of `atom_coords`
    """
    return squareform(pdist(atom_coords))


def make_transform_matrix(center, y=None, z=None):
    """Make 4x4 homogenous transformation matrix.

    Given Nx4 array A where A[:, 4] = 1., the transform matrix M should be
    used with dot(M, A.T).T. Order of operations is 1. translation, 2. align
    `y` x `z` plane to yz-plane 3. align `y` to y-axis.

    Parameters
    ----------
    center : 1x3 array of float
        Coordinate that should be centered after transformation.
    y : None or 1x3 array of float
        Vector that should lie on the y-axis after transformation
    z : None or 1x3 array of float
        Vector that after transformation should lie on yz-plane in direction
        of z-axis.

    Returns
    -------
    4x4 array of float
        4x4 homogenous transformation matrix.
    """
    translate = np.identity(4, dtype=np.float64)
    translate[:3, 3] = -np.asarray(center, dtype=np.float64)
    if y is not None:
        y = np.atleast_2d(y)
        if z is None:
            rotate = np.identity(4, dtype=np.float64)
            rotate[:3, :3] = make_rotation_matrix(y, Y_AXIS)
        else:
            z = np.atleast_2d(z)
            rotate_norm = np.identity(4, dtype=np.float64)
            x_unit = get_unit_vector(np.cross(y, z))
            rotate_norm[:3, :3] = make_rotation_matrix(x_unit, X_AXIS)
            new_y = np.dot(rotate_norm[:3, :3], y.flatten())
            rotate_y = np.identity(4, dtype=np.float64)
            rotate_y[:3, :3] = make_rotation_matrix(new_y.flatten(), Y_AXIS)
            rotate = np.dot(rotate_y, rotate_norm)
        transform = np.dot(rotate, translate)
    else:
        transform = translate
    return transform


def make_rotation_matrix(v0, v1):
    """Create 3x3 matrix of rotation from `v0` onto `v1`.

    Should be used by dot(R, v0.T).T.

    Parameters
    ----------
    v0 : 1x3 array of float
        Initial vector before alignment.
    v1 : 1x3 array of float
        Vector to which to align `v0`.
    """
    v0 = get_unit_vector(v0)
    v1 = get_unit_vector(v1)
    o = np.cross(v0, v1).flatten()
    if np.all(o == 0.):
        return np.identity(3, dtype=np.float64)
    u = get_unit_vector(o).flatten()
    sin_ang = np.linalg.norm(o)
    cos_ang = np.dot(v0, v1.T)
    ux = np.array([[ 0.  , -u[2],   u[1] ],
                   [ u[2],  0.  ,  -u[0] ],
                   [-u[1],  u[0],   0.   ]], dtype=np.float64)
    rot = (cos_ang * np.identity(3, dtype=np.float64) + sin_ang * ux +
           (1 - cos_ang) * np.outer(u, u))
    return rot


def transform_array(transform_matrix, array):
    """Pad an array with 1s, transform, and return with original dimensions.

    Parameters
    ----------
    transform_matrix : 4x4 array of float
        4x4 homogenous transformation matrix
    array : Nx3 array of float
        Array of 3-D coordinates.

    Returns
    -------
    Nx3 array of float : Transformed array
    """
    return unpad_array(np.dot(transform_matrix, pad_array(array).T).T)


def pad_array(arr, n=1., axis=1):
    """Return `arr` with row of `n` appended to `axis`.

    Parameters
    ----------
    arr : ndarray
        Array to pad
    n : float or int, optional
        Value to pad `arr` with
    axis : int, optional
        Axis of `arr` to pad with `n`.

    Returns
    -------
    ndarry
        Padded array.
    """
    if arr.ndim == 1:
        pad = np.ones(arr.shape[0]+1, dtype=arr.dtype) * n
        pad[:arr.shape[0]] = arr
    else:
        shape = list(arr.shape)
        shape[axis] += 1
        pad = np.ones(shape, dtype=arr.dtype)
        pad[:arr.shape[0], :arr.shape[1]] = arr
    return pad


def unpad_array(arr, axis=1):
    """Return `arr` with row removed along `axis`

    Parameters
    ----------
    arr : ndarry
        Array from which to remove row
    axis : int, optional
        Axis from which to remove row

    Returns
    -------
    ndarray
        Unpadded array.
    """
    if arr.ndim == 1:
        return arr[:-1]
    else:
        shape = list(arr.shape)
        shape[axis] -= 1
        return arr[:shape[0], :shape[1]]


def project_to_plane(vec_array, normal):
    """Project array of vectors to plane with normal `normal`.

    Parameters
    ----------
    vec_array : Nx3 array
        Array of N 3D vectors.
    normal : 1x3 array
        Normal vector to plane.

    Returns
    -------
    Nx3 array
        Array of vectors projected onto plane.
    """
    unit_normal = get_unit_vector(normal).flatten()
    mag_on_norm = np.dot(vec_array, unit_normal)
    if vec_array.ndim == 1:
        vec_on_norm = np.array(unit_normal, copy=True)
        vec_on_norm *= mag_on_norm
    else:
        vec_on_norm = np.tile(unit_normal, (vec_array.shape[0], 1))
        vec_on_norm *= mag_on_norm[:, None]
    return vec_array - vec_on_norm


def calculate_angles(vec_arr, ref, ref_norm=None):
    """Calculate angles between vectors in `vec_arr` and `ref` vector.

    If `ref_norm` is not provided, angle ranges between 0 and pi. If it is
    provided, angle ranges between 0 and 2pi. Note that if `ref_norm` is
    orthogonal to `vec_arr` and `ref`, then the angle is rotation around the
    axis, but if a non-orthogonal axis is provided, this may not be the case.

    Parameters
    ----------
    vec_arr : Nx3 float array
        Array of N 3D vectors.
    ref : 1x3 float array
        Reference vector
    ref_norm : 1x3 float array
        Normal vector.

    Returns
    -------
    1-D array
        Array of N angles
    """
    unit_vec_arr = get_unit_vector(vec_arr)
    unit_ref = get_unit_vector(ref).flatten()
    ang = np.arccos(np.clip(np.dot(unit_vec_arr, unit_ref), -1.0, 1.0))
    if ref_norm is not None:
        sign = np.sign(np.dot(ref_norm,
                              np.cross(unit_vec_arr, unit_ref).T)).flatten()
        sign[sign == 0] = 1
        ang = rotate_angles(sign * ang, 2 * np.pi)
    return ang


def rotate_angles(angles, amount):
    """Rotate angles by `amount`, keeping in 0 to 2pi range.

    Parameters
    ----------
    angles : 1-D float array
        Angles in radians
    amount : Amount to rotate by
        Amount to rotate angles by

    Returns
    -------
    1-D float array : Rotated angles
    """
    return (angles + amount) % (2 * np.pi)


def shell_to_pdb(mol, shell, atom_coords, bound_atoms_dict, out_file=None,
                 reorient=True):
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
    header_lines = [remark+" COMPOUND", remark+" "+mol.GetProp("_Name")]
    lines = header_lines + ["MODEL", ]
    atom_ids = sorted(shell.substruct.atoms)
    atoms = map(mol.GetAtomWithIdx, atom_ids)
    coords = np.asarray(map(atom_coords.get, atom_ids), dtype=np.float64)
    if reorient:
        try:
            coords = transform_array(shell.transform_matrix, coords)
        except AttributeError:
            coords -= atom_coords[shell.center_atom]

    for i, atom_id in enumerate(atom_ids):
        elem = atoms[i].GetSymbol()
        name = "{}{:d}".format(elem, atom_id)
        charge = atoms[i].GetFormalCharge()
        if charge > 0:
            charge = "{:d}+".format(charge)
        elif charge < 0:
            charge = "{:d}-".format(abs(charge))
        else:
            charge = ""
        if atom_id == shell.center_atom:
            temp = 1.
        elif atom_id in shell.atoms:
            temp = .5
        else:
            temp = 0.
        pdb_entries = {"atom_id": atom_id,
                       "name": name,
                       "coord": coords[i, :].flatten(),
                       "occupancy": 0.,
                       "temp": temp,
                       "elem": elem,
                       "charge": charge}
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

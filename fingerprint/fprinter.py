"""Tools for generating E3FP Fingerprints.

Author: Seth Axen
E-mail: seth.axen@gmail.com"""
from __future__ import division, print_function

import numpy as np
from scipy.spatial.distance import pdist, squareform

from rdkit import Chem
import mmh3

from e3fp.fingerprint.fprint import Fingerprint, CountFingerprint


class Fingerprinter(object):

    """E3FP fingerprint generator.

    Description
    -----------
    ECFP builds increasingly larger *connected* substructures centered around
    each atom, where each iteration expands the substructure by one bond in
    each direction. E3FP uses spherical shells around each atom to define
    substructure; as a result, atoms may be included in substructure which are
    not directly connected to any other atoms in the substructure. Through
    this subtle but important modification, E3FP fingerprints represent
    specific conformers of a molecule.

    Parameters
    ----------
    bits : int, optional (default 32)
        Bits of integer indices to hash arrays to. Result is `bits`-bit sparse
        indices for on bits in 2^`bits` length bitvector.
    level : int or None, optional (default None)
        Maximum number of iterations for fingerprint generation. If None, runs
        until no new substructures are identified. Because this could produce
        a different final level number for each conformer, it's recommended to
        choose a terminal level number.
    radius_multiplier : float, optional (default 2.0)
        Multiple by which to increase shell size. At iteration 0, shell radius
        is 0*`radius_multiplier`, at iteration 2, radius is
        2*`radius_multiplier`, etc.
    counts : bool, optional (default False)
        Instead of simple ``Fingerprint`` object, generates
        ``CountFingerprint`` that tracks number of times each bit appears in a
        fingerprint.
    stereo : bool, optional (default False)
        Differentiate based on stereochemistry. Resulting fingerprints are not
        comparable to non-stereo fingerprints.
    stereo_cutoff : float, optional (default pi/36)
        Stereo-sensitive fingerprints use dihedral angles
    merge_duplicate_substructs : bool, optional (default True)
        If a substructure ever arises that has been seen before, corresponding
        identifier is not added to sparse indices.
    store_identifiers_map : bool, optional (default False)
        Store a dictionary in the fingerprint mapping identifiers to
        substructures as defined by a tuple containing iteration number,
        center atom index, and a tuple of member atoms in the sphere.
    store_identifier_id_map : bool, optional (default False)
        Store a dictionary in the fingerprint that maps from each on bit to a
        set of the atom indices at the centers of the corresponding
        substructures.
    include_disconnected : bool, optional (default True):
        Include disconnected atoms from hashes and substructure. E3FP's
        advantage over ECFP relies on disconnected atoms, so the option to
        turn this off is present only for testing.
    """

    def __init__(self, bits=32, level=None, radius_multiplier=2.0,
                 stereo=False, stereo_cutoff=np.pi / 18, counts=False,
                 merge_duplicate_substructs=True, store_identifiers_map=False,
                 store_identifier_id_map=False, include_disconnected=True):
        """Initialize fingerprinter settings."""
        self.mol = None
        if level is not None:
            self.level = level
        else:
            self.level = np.inf
        if not np.log2(bits).is_integer():
            raise ValueError("bits are not multiple of 32")
        self.bits = bits
        self.radius_multiplier = radius_multiplier
        if counts:
            self.fp_type = CountFingerprint
        else:
            self.fp_type = Fingerprint
        self.stereo = stereo
        self.stereo_cutoff = stereo_cutoff
        self.merge_duplicate_substructs = merge_duplicate_substructs
        self.store_identifiers_map = store_identifiers_map
        self.store_identifier_id_map = store_identifier_id_map
        self.include_disconnected = include_disconnected
        self.bond_types = {None: 5,
                           Chem.BondType.SINGLE: 1,
                           Chem.BondType.DOUBLE: 2,
                           Chem.BondType.TRIPLE: 3,
                           Chem.BondType.AROMATIC: 4}
        self.reset()

    def run(self, conf=None, mol=None, conf_id=None, return_substruct=False):
        """Generate fingerprint from provided conformer or mol and conf id.

        Parameters
        ----------
        conf : RDKit Conformer, optional (default None)
            Input conformer. If this option is provided, `mol` and `conf_id`
            will be ignored.
        mol : RDKit Mol, optional (default None)
            Input molecule object, with at least one conformer. If `conf_id`
            is also specified, that conformer is considered. Otherwise, first
            conformer is used.
        conf_id : int, optional (default None)
            Integer ID of conformer in `mol`
        return_substruct : bool, optional (default False)
            Return dict mapping substructure to fingerprint indices. Keys are
            indices, values are list of substructures, represented as a tuple
            of atom indices where the first index is the central atom and the
            remaining indices (within the sphere) are sorted.
        """
        if conf is not None:
            mol = conf.GetOwningMol()

        if mol is not None and mol is not self.mol:
            self.reset_mol()
            self.mol = mol
            self.initialize_mol(mol)
        if conf is None:
            if conf_id is None or conf_id < 0:
                conf_id = 0
            conf = self.mol.GetConformers()[conf_id]

        self.conf = conf

        self.initialize_conformer()
        self.atom_identifiers = {0: self.init_identifiers}
        self.generate_fingerprints()

    def reset(self):
        """Clear all variables associated with the last run."""
        self.reset_mol()

    def reset_conf(self):
        """Clear only conformer-specific variables."""
        self.conformers = None
        self.conf_ids = None
        self.neighbor_dict = {}
        self.atom_coords = None
        self.all_identifiers = []
        self.all_identifier_substructs = []
        self.all_identifier_ids = []
        self.atom_identifiers = {}
        self.indices_to_substructs = {}
        self.substructs_to_indices = {}
        self.identifiers_at_level = {}
        self.fingerprints = {}
        self.rng = np.random.RandomState(42)

    def reset_mol(self):
        """Clear all variables associated with the molecule."""
        self.atoms = None
        self.atom_ids = None
        self.atom_indices = None
        self.id_to_index = None
        self.index_to_id = None
        self.init_identifiers = None
        self.bonds = {}
        self.atom_bonds = {}
        self.reset_conf()

    def initialize_mol(self, mol):
        """Set general properties of `mol` that apply to all its conformers.

        Parameters
        ----------
        mol : RDKit Mol
            Input molecule ``Mol`` object.
        """
        self.mol = mol
        self.atoms = np.array([x for x in mol.GetAtoms()
                               if x.GetAtomicNum() > 1])  # ignore hydrogens
        self.atom_ids = np.array([x.GetIdx()
                                  for x in self.atoms], dtype=np.int)
        self.atom_indices = np.arange(self.atoms.shape[0], dtype=np.int)

        # atom_ids don't necessarily match atom_indices, so we need a map
        self.id_to_index = dict(zip(self.atom_ids, self.atom_indices))
        self.index_to_id = dict(zip(self.atom_indices, self.atom_ids))

        self.initialize_identifiers()
        bonds_ids = dict([((x.GetBeginAtomIdx(), x.GetEndAtomIdx()),
                           self.bond_types[x.GetBondType()])
                          for x in self.mol.GetBonds()])

        self.bonds = dict([((self.id_to_index[k[0]],
                             self.id_to_index[k[1]]), v)
                           for k, v in bonds_ids.items()
                           if (k[0] in self.id_to_index and
                               k[1] in self.id_to_index)])
        self.bonds.update([((k[1], k[0]), v) for k, v in self.bonds.items()])
        self.atom_bonds = {}
        _ = [self.atom_bonds.setdefault(x[0], set()).add(x[1])
             for x in self.bonds.keys()]
        del _

    def initialize_identifiers(self):
        """Set initial identifiers for atoms."""
        self.init_identifiers = self.initialize_ecfp_identifiers(self.atoms)

    @classmethod
    def initialize_ecfp_identifiers(cls, atoms, bits=32):
        """Initialize ids according to Daylight invariants.

        Parameters
        ----------
        atoms : ndarray of ``RDKit Atom``
            List of heavy atoms in molecule
        bits : int, optional (default 32)
            Number of bits of integer IDs.

        Returns
        -------
        ndarray of int : initial ids for atoms
        """
        # Similar to rdkit's invariants, but rdkit combines the first two with
        # just atom.getTotalDegree()
        table = Chem.rdchem.GetPeriodicTable()
        identifier_array = [cls.hash_int_array(
            np.array([x.GetDegree(),
                      x.GetTotalValence() - x.GetTotalNumHs(),
                      x.GetAtomicNum(),
                      x.GetMass() - table.GetAtomicWeight(x.GetAtomicNum()),
                      x.GetFormalCharge(),
                      x.GetTotalNumHs(),
                      x.IsInRing()],
                     dtype=np.int), bits=bits, positive=True)
            for x in atoms]
        return np.array(identifier_array, dtype=np.long)

    def get_fingerprint_at_level(self, level=-1, atom_ids_mask=set()):
        """Retrieve all conformer fingerprints at specified level.

        Parameters
        ----------
        level : int, optional (default -1)
            Level of fingerprints retrieved. -1 or None retrieves highest
            level common to all fingerprints.

        Returns
        -------
        Fingerprint or bool : ``Fingerprint`` at specified level. If no
                              fingerprint exists at that level, None is
                              returned.
        """
        if isinstance(atom_ids_mask, int):
            atom_ids_mask = {atom_ids_mask}
        self.indices_mask = {self.id_to_index[x] for x in list(atom_ids_mask)
                             if x in self.id_to_index}

        if level in self.identifiers_at_level:
            true_level = level
        elif level == -1 or level is None or level <= self.level:
            true_level = max(self.identifiers_at_level.keys())
        else:
            return None

        if len(self.indices_mask) == 0:
            masked_identifiers = self.identifiers_at_level[true_level]
        else:
            masked_identifiers = [
                x for i, x in enumerate(self.identifiers_at_level[true_level])
                if self.all_identifier_substructs[i].isdisjoint(
                    self.indices_mask)]

        fingerprint = self.fp_type(masked_identifiers, level=level,
                                   bits=2**self.bits)

        if self.store_identifier_id_map:
            fingerprint.index_id_map = self.get_identifier_id_map(true_level)

        return fingerprint

    def get_identifier_id_map(self, level=-1):
        max_level = max(self.identifiers_at_level.keys())
        if level == -1 or level is None or level > max_level:
            level = max_level
        identifier_id_map = {}
        for i, index in enumerate(self.identifiers_at_level[level]):
            identifier_id_map.setdefault(index, set()).add(
                self.all_identifier_ids[i])
        return identifier_id_map

    def initialize_conformer(self):
        """Retrieve atom coordinates and build dict of atom neighbors."""
        self.atom_coords = np.array([self.conf.GetAtomPosition(x)
                                     for x in self.atom_ids],
                                    dtype=np.float)
        distance_matrix = self.build_distance_matrix(self.atom_coords)
        self.build_neighbor_dict(distance_matrix)
        self.build_substructs()

    @staticmethod
    def build_distance_matrix(atom_coords):
        """Build pairwise distance matrix from conformer coordinates.

        Parameters
        ----------
        atom_coords : ndarray of float
            an Mx3 array of x,y,z atom coordinates.

        Returns
        -------
        ndarray of float : square symmetrical distance matrix with indices
                           corresponding to indices of `atom_coords`

        """
        return squareform(pdist(atom_coords))

    def build_neighbor_dict(self, distance_matrix):
        """Determine neighbor atoms within each atom's shell at each
        iteration.

        Parameters
        ----------
        distance_matrix : ndarray of float
            Square symmetric matrix of pairwise atom distances with dimensions
            MxM.
        """
        self.neighbor_dict = {0: dict([(x, frozenset((x,)))
                                       for x in self.atom_indices])}
        bonds_with_center_dict = dict([(x[0], frozenset().union(x[1], (x[0],)))
                                       for x in self.atom_bonds.iteritems()])

        i = 1
        while i <= self.level:
            rad = i * self.radius_multiplier
            atom_pairs = np.where(distance_matrix <= rad)
            if not self.include_disconnected:
                self.neighbor_dict[i] = bonds_with_center_dict
            else:
                matches = dict([(j, frozenset(
                    atom_pairs[1][np.where(atom_pairs[0] == j)]))
                    for j in self.atom_indices])
                self.neighbor_dict[i] = matches
            if atom_pairs[0].shape[0] == self.atoms.shape[0]**2:
                # all possible pairs are within same shell
                break
            i += 1

    def build_substructs(self):
        """Build a dict of substructures.

        Substructure is represented as a ``frozenset`` of all atom indices
        that contributed information to a given atom at a given level. It is
        the union of that atom, its neighbors at a current level, and the
        substructures of those neighbors and itself at the previous level.
        """
        self.indices_to_substructs = {}
        self.substructs_to_indices = {}

        for level in sorted(self.neighbor_dict.keys()):
            if level == 0:
                self.indices_to_substructs[level] = self.neighbor_dict[level]
            else:
                for atom_index in self.atom_indices:
                    substruct = reduce(frozenset().union, (
                        self.indices_to_substructs[level-1][x]
                        for x in self.neighbor_dict[level][atom_index]))
                    self.indices_to_substructs.setdefault(
                        level, {})[atom_index] = substruct
                    self.substructs_to_indices.setdefault(
                        level, {}).setdefault(
                        substruct, set([])).add(atom_index)

    def generate_fingerprints(self):
        """Generate a fingerprint for a conformer."""
        initial_identifiers = self.atom_identifiers[0]

        self.all_identifiers = list(initial_identifiers)
        self.all_identifier_substructs = [self.indices_to_substructs[0][x]
                                          for x in self.atom_indices]
        self.all_identifier_ids = map(self.index_to_id.get, self.atom_indices)

        self.identifiers_at_level = {}
        self.identifiers_at_level[0] = list(self.all_identifiers)

        past_substructs_set = set(self.indices_to_substructs[0].values())

        level = 1  # iteration/level number
        while level in self.neighbor_dict:
            new_atom_identifiers = np.empty_like(
                self.atom_identifiers[level - 1], dtype=np.long)

            for atom_index in self.atom_indices:
                new_atom_identifier = self.calculate_atom_identifier(
                    atom_index, level)
                new_atom_identifiers[atom_index] = new_atom_identifier

            self.atom_identifiers[level] = new_atom_identifiers

            # filter out any identifiers based on substructure
            if self.merge_duplicate_substructs:
                (accepted_indices,
                 accepted_identifiers) = self.filter_identifiers(
                    new_atom_identifiers, past_substructs_set,
                    self.substructs_to_indices[level])
            else:
                accepted_indices = self.indices
                accepted_identifiers = new_atom_identifiers

            self.all_identifiers.extend(accepted_identifiers)
            self.all_identifier_substructs.extend(
                [self.indices_to_substructs[level][x]
                 for x in accepted_indices])
            self.all_identifier_ids.extend(
                map(self.index_to_id.get, accepted_indices))

            past_substructs_set.update(
                *self.indices_to_substructs[level].values())

            self.identifiers_at_level[level] = self.all_identifiers
            level += 1

    def calculate_atom_identifier(self, atom_index, level):
        """Determine new identifier for an atom at a specific level.

        Parameters
        ----------
        atom_index : int
            Index of central atom.
        level : int
            Level, i.e. iteration number.

        Returns
        -------
        long : New identifier for atom, corresponds to sparse index in
               fingerprint.
        """
        neighbor_indices = self.neighbor_dict[level][atom_index].difference(
            (atom_index,))

        neighbor_tuples_list = [
            (self.bonds.get((atom_index, x), self.bond_types[None]),
             self.atom_identifiers[level - 1][x], x)
            for x in neighbor_indices]

        sorted_neighbors = sorted(neighbor_tuples_list, key=lambda x: x[:2])

        if self.stereo:
            data_list_2d = self.add_stereo_indicator(atom_index,
                                                     sorted_neighbors)
        else:
            data_list_2d = [x[:2] for x in sorted_neighbors]

        try:
            data_array_len = (
                len(data_list_2d) * len(data_list_2d[0]) + 2)
        except IndexError:  # raised when sorted_neighbors empty
            data_array_len = 2

        # data_array is 1-D array which is hashed to identifier
        data_array = np.empty((data_array_len,), dtype=np.long)
        data_array[0] = level
        data_array[1] = self.atom_identifiers[level - 1][atom_index]
        data_array[2:] = [y for x in data_list_2d for y in x]
        new_atom_identifier = self.hash_int_array(data_array,
                                                  bits=self.bits,
                                                  positive=True)
        return new_atom_identifier

    def add_stereo_indicator(self, atom_index, sorted_neighbors):
        """Add an indicator for relative 3D orientation of atoms to data.

        Vectors from the center atom to each neighbor atom are projected on
        the unit sphere. The first unique atom by identifier is selected, if
        possible. If none can be selected, the average vector of all
        neighbors is calculated and projected to the unit sphere. The
        `y`-axis is set to this vector. The angle between each unit vector
        and the `y`-axis is calculated, and the atom closest to pi/2 rad from
        the north pole with an identifier unique to an atom at that angle is
        used to define the direction of the `z`-axis around the `y`-axis.
        Indicators (`s`) are then assigned. The atoms in the :math:`y\geq0`
        and :math:`y<0` hemispheres have positive and negative indicators,
        respectively. :math:`|s|=1` is assigned to atoms whose unit vectors
        fall within pi/36 rad (5 deg) radians of the pole. The remaining
        surface of the unit sphere is divided into eight sections, four in
        each hemisphere. `z` falls in :math:`|s|=2`, indicators +/- 3-5 are
        assigned to remaining quadrants radially around the `y`-axis. If two
        unique atoms could not be chosen, all atoms are assigned indicators
        of 0.

        Parameters
        ----------
        atom_index : int
            Index of central atom.
        sorted_neighbors : list of tuple
            Each ``tuple`` corresponds to a neighboring atom in the sphere.
            The ``tuple`` consists of an ``int`` indicating type of bond
            between the central atom and its neighbor, a ``long`` indicating
            that neighbor's identifier at the last level, and the index of
            the neighbor. This list is assumed to be sorted.

        Returns
        -------
        list of tuple : A 2D data list similar in structure to
                        `sorted_neighbors`. The neighbor index in each tuple
                        is replaced with a stereo indicator, and the list is
                        re-sorted.
        """
        if len(sorted_neighbors) == 0:
            return []

        (sorted_neighbor_atom_types,
         sorted_neighbor_identifiers,
         sorted_neighbor_indices) = zip(*sorted_neighbors)

        data_list_2d = []
        stereo_indicators = np.zeros((len(sorted_neighbors),), dtype=np.int)

        if stereo_indicators.shape[0] > 1:
            # project atom vectors onto unit sphere
            atom_units = self.get_unit_vector(
                self.atom_coords[sorted_neighbor_indices, :] -
                self.atom_coords[[atom_index], :])

            y_inds = self.get_first_unique_tuple_inds(sorted_neighbors, 1)

            # select y-axis vector
            if len(y_inds) > 0:  # unique atom could be found
                y_coord = (
                    self.atom_coords[[sorted_neighbor_indices[y_inds[0]]], :])
                y_unit = atom_units[y_inds, :]
                stereo_indicators[y_inds[0]] = -1
            else:  # no unique atom could be found
                # assign y-axis as mean vector
                y_coord = np.mean(self.atom_coords[sorted_neighbor_indices, :],
                                  axis=0)
                y_unit = self.get_unit_vector(
                    y_coord - self.atom_coords[[atom_index], :])

            # calculate angle between y-axis and each unit vector
            long_angle = np.arccos(np.clip(
                np.dot(atom_units, y_unit.flatten()), -1.0, 1.0))

            # get first unique atom based on closeness to pi/2 from y-axis
            z_inds = self.get_first_unique_tuple_inds(
                sorted(zip(np.abs(np.pi/2 - np.abs(long_angle)),
                           sorted_neighbor_identifiers)), 1, ignore=y_inds)

            if len(z_inds) > 0:
                atom_lat_units = self.get_unit_vector(
                    np.cross(np.cross(y_unit, atom_units), y_unit))

                pos_angle_from_z = np.arccos(
                    np.clip(np.dot(atom_lat_units,
                            atom_lat_units[z_inds[0]]), -1.0, 1.0))

                angle_from_z = pos_angle_from_z * np.sign(
                    np.dot(y_unit, np.cross(atom_lat_units,
                                            atom_lat_units[z_inds[0]]).T))

                # offset by pi/4 so z-axis isn't an edge case
                lat_angle = (angle_from_z + 2*np.pi + np.pi/4) % (2*np.pi)
                lat_indicators = np.asarray(lat_angle * 4 / (2*np.pi),
                                            dtype=np.int) + 2

                stereo_indicators = np.asarray(
                    np.sign(long_angle - np.pi) * lat_indicators,
                    dtype=np.int).flatten()

                stereo_indicators[np.where(long_angle < np.pi/36)] = -1
                stereo_indicators[np.where(
                    long_angle + (np.pi/36) > np.pi)] = 1

        # re-sort
        data_list_2d = sorted(zip(sorted_neighbor_atom_types,
                                  sorted_neighbor_identifiers,
                                  stereo_indicators.flatten()))
        return data_list_2d

    @staticmethod
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

    def filter_identifiers(self, atom_identifiers, past_substructs_set,
                           substructs_to_indices):
        """Given atom identifiers, return filtered identifiers.

        Parameters
        ----------
        atom_identifiers : ndarray of int
            Array of atom identifiers. Indices of array correspond to
            `atom_indices`.
        past_substructs_set : set of tuples
            Set of existing substructures, represented by tuples of
            `atom_indices`.
        substructs_to_indices : dict
            New substructures, where keys are substructure tuples and values
            are a list of tuples of atom identifier and `atom_indices`. Each
            of these tuples corresponds to an atom that is the center of that
            substructure in this iteration.

        Returns
        -------
        ndarray of int : Accepted atom indices
        ndarray of int : Filtered atom identifiers
        """
        accepted_indices = np.zeros_like(atom_identifiers, dtype=np.bool_)

        novel_substructs = [sorted(vs, key=lambda x: atom_identifiers[x])[0]
                            for s, vs in substructs_to_indices.iteritems()
                            if s not in past_substructs_set]
        accepted_indices[novel_substructs] = True

        return (self.atom_ids[accepted_indices],
                atom_identifiers[accepted_indices])

    def get_identifier_substructs_map(self):
        """Return map from identifier to substructure.

        Returns
        -------
        dict : Dictionary mapping identifiers (sparse indices) to
               substructure.
        """
        return self.identifier_indices_to_substructs

    @classmethod
    def get_length_vector(cls, v, n=1.):
        """Return vector of length `n` parallel to vector `v`.

        Parameters
        ----------
        v : ndarray of float
            Nx3 vector.

        Returns
        -------
        ndarray of float : Vector of length `n`
        """
        mag = np.linalg.norm(v, axis=1).reshape(v.shape[0], 1)
        mag[np.where(mag == 0.)] = 1.

        return n * v / mag

    @classmethod
    def get_unit_vector(cls, v):
        """Return unit vector parallel to vector `v`.

        Parameters
        ----------
        v : ndarray of float
            Nx3 vector.

        Returns
        -------
        ndarray of float : Unit vector of `v`, i.e. `v` divided by its
                           magnitude.
        """
        return cls.get_length_vector(v, n=1.)

    @staticmethod
    def hash_int_array(array, bits=32, positive=False):
        """Hash an integer array into a integer with the specified bits.

        Parameters
        ----------
        array : ndarray of int
            Numpy array containing integers
        bits : int, optional (default 32)
            Number of bits of resulting integer; valid options are 32, 64, and
            128.
        positive : bool, optional (default False)
            Convert integer to positive

        Returns
        -------
        int : 32-bit integer
        """
        if bits == 32:
            hash_value = mmh3.hash(array) % 2**32
        elif bits == 64:
            hash_value = mmh3.hash64(array) % 2**64
        elif bits == 128:
            hash_value = mmh3.hash128(array) % 2**128
        else:
            raise ValueError(
                "Bits must be 32, 64, or 128, not %s." % str(bits))

        if positive:
            # ensures all hashed integers are positive
            return (hash_value + 2**32) % 2**32
        else:
            return hash_value

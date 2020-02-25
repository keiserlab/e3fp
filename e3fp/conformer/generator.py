"""Conformer generation.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import logging

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PropertyMol
from .util import add_conformer_energies_to_mol

# Heavily modified by Seth Axen from code under the following license
__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

# options
FORCEFIELD_CHOICES = ("uff", "mmff94", "mmff94s")

# default values
NUM_CONF_DEF = -1
FIRST_DEF = -1
POOL_MULTIPLIER_DEF = 1
RMSD_CUTOFF_DEF = 0.5
MAX_ENERGY_DIFF_DEF = -1.0
FORCEFIELD_DEF = "uff"
SEED_DEF = -1


class ConformerGenerator(object):
    """Generate conformers using RDKit.

    Procedure
    ---------
    1. Generate a pool of conformers.
    2. Minimize conformers.
    3. Filter conformers using an RMSD threshold and optional minimum energy
       difference.

    Note that pruning is done _after_ minimization, which differs from the
    protocol described in the references.

    References
    ----------
    * http://rdkit.org/docs/GettingStartedInPython.html
      #working-with-3d-molecules
    * http://pubs.acs.org/doi/full/10.1021/ci2004658
    * https://github.com/skearnes/rdkit-utils/blob/master/rdkit_utils/
      conformers.py
    """

    def __init__(
        self,
        num_conf=NUM_CONF_DEF,
        first=FIRST_DEF,
        rmsd_cutoff=RMSD_CUTOFF_DEF,
        max_energy_diff=MAX_ENERGY_DIFF_DEF,
        forcefield=FORCEFIELD_DEF,
        pool_multiplier=POOL_MULTIPLIER_DEF,
        seed=SEED_DEF,
        get_values=False,
        sparse_rmsd=True,
        store_energies=True,
    ):
        """Initialize generator settings.

        Parameters
        ----------
        num_conf : int, optional
            Maximum number of conformers to generate (after pruning). -1
            results in auto selection of max_conformers.
        first : int, optional
            Terminate when this number of conformers has been accepted, and
            only return those conformers.
        pool_multiplier : int, optional
            Factor to multiply by max_conformers to generate the initial
            conformer pool. Since conformers are filtered after energy
            minimization, increasing the size of the pool increases the chance
            of identifying max_conformers unique conformers.
        rmsd_cutoff : float, optional
            RMSD cutoff for pruning conformers. If None or negative, no
            pruning is performed.
        max_energy_diff : float, optional
            If set, conformers with energies this amount above the minimum
            energy conformer are not accepted.
        forcefield : {'uff', 'mmff94', 'mmff94s'}, optional
            Force field to use for conformer energy calculation and
            minimization.
        seed : int, optional
            Random seed for conformer generation. If -1, the random number
            generator is unseeded.
        get_values : boolean, optional
            Return tuple of key values, for storage.
        sparse_rmsd : bool, optional
            If `get_values` is True, instead of returning full symmetric RMSD
            matrix, only return flattened upper triangle.
        store_energies : bool, optional
            Store conformer energies as property in mol.
        """
        self.max_conformers = num_conf
        self.first_conformers = first
        if not rmsd_cutoff or rmsd_cutoff < 0:
            rmsd_cutoff = -1.0
        self.rmsd_cutoff = rmsd_cutoff

        if max_energy_diff is None or max_energy_diff < 0:
            max_energy_diff = -1.0
        self.max_energy_diff = max_energy_diff

        if forcefield not in FORCEFIELD_CHOICES:
            raise ValueError(
                "%s is not a valid option for forcefield" % forcefield
            )
        self.forcefield = forcefield
        self.pool_multiplier = pool_multiplier
        self.seed = seed
        self.get_values = get_values
        self.sparse_rmsd = sparse_rmsd
        self.store_energies = store_energies

    def __call__(self, mol):
        """Generate conformers for a molecule.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.

        Returns
        -------
        RDKit Mol : copy of the input molecule with embedded conformers
        """
        return self.generate_conformers(mol)

    def generate_conformers(self, mol):
        """Generate conformers for a molecule.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.

        Returns
        -------
        RDKit Mol : copy of the input molecule with embedded conformers
        """
        # initial embedding
        mol = self.embed_molecule(mol)
        if not mol.GetNumConformers():
            msg = "No conformers generated for molecule"
            if mol.HasProp("_Name"):
                name = mol.GetProp("_Name")
                msg += ' "{}".'.format(name)
            else:
                msg += "."
            raise RuntimeError(msg)

        # minimization and filtering
        self.minimize_conformers(mol)
        mol, indices, energies, rmsds = self.filter_conformers(mol)

        if self.store_energies:
            add_conformer_energies_to_mol(mol, energies)

        if self.get_values is True:
            if self.sparse_rmsd:
                rmsds_mat = rmsds[np.triu_indices_from(rmsds, k=1)]
            else:
                rmsds_mat = rmsds
            return mol, (self.max_conformers, indices, energies, rmsds_mat)
        else:
            return mol

    @staticmethod
    def get_num_conformers(mol):
        """Return ideal number of conformers from rotatable bond number in model.

        Parameters
        ----------
        mol : Mol
            RDKit `Mol` object for molecule

        Yields
        ------
        num_conf : int
            Target number of conformers to accept
        """
        num_rot = AllChem.CalcNumRotatableBonds(mol)
        if num_rot < 8:
            return 50
        elif num_rot >= 8 and num_rot <= 12:
            return 200
        elif num_rot > 12:
            return 300
        else:
            return 0

    def embed_molecule(self, mol):
        """Generate conformers, possibly with pruning.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.
        """
        logging.debug("Adding hydrogens for %s" % mol.GetProp("_Name"))
        mol = Chem.AddHs(mol)  # add hydrogens
        logging.debug("Hydrogens added to %s" % mol.GetProp("_Name"))
        logging.debug("Sanitizing mol for %s" % mol.GetProp("_Name"))
        Chem.SanitizeMol(mol)
        logging.debug("Mol sanitized for %s" % mol.GetProp("_Name"))
        if self.max_conformers == -1 or type(self.max_conformers) is not int:
            self.max_conformers = self.get_num_conformers(mol)
        n_confs = self.max_conformers * self.pool_multiplier
        if self.first_conformers == -1:
            self.first_conformers = self.max_conformers
        logging.debug(
            "Embedding %d conformers for %s" % (n_confs, mol.GetProp("_Name"))
        )
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=n_confs,
            maxAttempts=10 * n_confs,
            pruneRmsThresh=-1.0,
            randomSeed=self.seed,
            ignoreSmoothingFailures=True,
        )
        logging.debug("Conformers embedded for %s" % mol.GetProp("_Name"))
        return mol

    def get_molecule_force_field(self, mol, conf_id=None, **kwargs):
        """Get a force field for a molecule.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.
        conf_id : int, optional
            ID of the conformer to associate with the force field.
        **kwargs : dict, optional
            Keyword arguments for force field constructor.
        """
        if self.forcefield == "uff":
            ff = AllChem.UFFGetMoleculeForceField(
                mol, confId=conf_id, **kwargs
            )
        elif self.forcefield.startswith("mmff"):
            AllChem.MMFFSanitizeMolecule(mol)
            mmff_props = AllChem.MMFFGetMoleculeProperties(
                mol, mmffVariant=self.forcefield
            )
            ff = AllChem.MMFFGetMoleculeForceField(
                mol, mmff_props, confId=conf_id, **kwargs
            )
        else:
            raise ValueError(
                "Invalid forcefield " + "'{}'.".format(self.forcefield)
            )
        return ff

    def minimize_conformers(self, mol):
        """Minimize molecule conformers.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.
        """
        logging.debug("Minimizing conformers for %s" % mol.GetProp("_Name"))
        for conf in mol.GetConformers():
            ff = self.get_molecule_force_field(mol, conf_id=conf.GetId())
            ff.Minimize()
        logging.debug("Conformers minimized for %s" % mol.GetProp("_Name"))

    def get_conformer_energies(self, mol):
        """Calculate conformer energies.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.

        Returns
        -------
        energies : array_like
            Minimized conformer energies.
        """
        num_conf = mol.GetNumConformers()
        energies = np.empty((num_conf,), dtype=np.float)
        for i, conf in enumerate(mol.GetConformers()):
            ff = self.get_molecule_force_field(mol, conf_id=conf.GetId())
            energies[i] = ff.CalcEnergy()
        return energies

    def filter_conformers(self, mol):
        """Filter conformers which do not meet an RMSD threshold.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.

        Returns
        -------
        A new RDKit Mol containing the chosen conformers, sorted by
        increasing energy.
        """
        logging.debug("Pruning conformers for %s" % mol.GetProp("_Name"))
        energies = self.get_conformer_energies(mol)
        energy_below_threshold = np.ones_like(energies, dtype=np.bool_)

        sort = np.argsort(energies)  # sort by increasing energy
        confs = np.array(mol.GetConformers())

        # remove hydrogens to speed up substruct match
        mol = Chem.RemoveHs(mol)
        accepted = []  # always accept lowest-energy conformer
        rejected = []
        rmsds = np.zeros((confs.shape[0], confs.shape[0]), dtype=np.float)
        for i, fit_ind in enumerate(sort):
            accepted_num = len(accepted)

            # always accept lowest-energy conformer
            if accepted_num == 0:
                accepted.append(fit_ind)

                # pre-compute if Es are in acceptable range of min E
                if self.max_energy_diff != -1.0:
                    energy_below_threshold = (
                        energies <= energies[fit_ind] + self.max_energy_diff
                    )

                continue

            # reject conformers after first_conformers is reached
            if accepted_num >= self.first_conformers:
                rejected.append(fit_ind)
                continue

            # check if energy is too high
            if not energy_below_threshold[fit_ind]:
                rejected.append(fit_ind)
                continue

            # get RMSD to selected conformers
            these_rmsds = np.zeros((accepted_num,), dtype=np.float)
            # reverse so all confs aligned to lowest energy
            for j, accepted_ind in self.reverse_enumerate(accepted):
                this_rmsd = AllChem.GetBestRMS(
                    mol,
                    mol,
                    confs[accepted_ind].GetId(),
                    confs[fit_ind].GetId(),
                )
                # reject conformers within the RMSD threshold
                if this_rmsd < self.rmsd_cutoff:
                    rejected.append(fit_ind)
                    break
                else:
                    these_rmsds[-j - 1] = this_rmsd
            else:
                rmsds[fit_ind, accepted] = these_rmsds
                rmsds[accepted, fit_ind] = these_rmsds
                accepted.append(fit_ind)

        # slice and order rmsds and energies to match accepted list
        rmsds = rmsds[np.ix_(accepted, accepted)]
        energies = energies[accepted]

        # create a new molecule with all conformers, sorted by energy
        new = PropertyMol.PropertyMol(mol)
        new.RemoveAllConformers()
        conf_ids = [conf.GetId() for conf in mol.GetConformers()]
        for i in accepted:
            conf = mol.GetConformer(conf_ids[i])
            new.AddConformer(conf, assignId=True)

        logging.debug("Conformers filtered for %s" % mol.GetProp("_Name"))
        return new, np.asarray(accepted, dtype=np.int), energies, rmsds

    @staticmethod
    def reverse_enumerate(iterable):
        """Enumerate, but with the last result first but still numbered last.

        Parameters
        ----------
        iterable : some 1-D iterable

        Returns
        -------
        iterable:
            Reverse of `enumerate` function
        """
        return zip(reversed(range(len(iterable))), reversed(iterable))

    # magic methods
    def __repr__(self):
        return """ConformerGenerator(num_conf=%r, first=%r,\
               \n                   pool_multiplier=%r, rmsd_cutoff=%r,\
               \n                   max_energy_diff=%r, forcefield=%r,\
               \n                   get_values=%r, sparse_rmsd=%r)""" % (
            self.max_conformers,
            self.first,
            self.pool_multiplier,
            self.rmsd_cutoff,
            self.max_energy_diff,
            self.forcefield,
            self.get_values,
            self.sparse_rmsd,
        )

    def __str__(self):
        return self.__repr__()

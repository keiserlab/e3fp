"""Tests for conformer generation.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""

import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SDF_FILE_COMPRESSED = os.path.join(DATA_DIR, "ritalin_nonplanar.sdf.bz2")
SDF_FILE_UNCOMPRESSED = os.path.join(DATA_DIR, "ritalin_nonplanar.sdf")

class TestConformer:
    def test_standardisation(self):
        import rdkit.Chem
        from e3fp.conformer.util import (
            mol_from_smiles,
            mol_to_standardised_mol,
        )

        smiles = "C[N-]c1cccc[n+]1C"
        mol = mol_from_smiles(smiles, "tmp")
        assert rdkit.Chem.MolToSmiles(mol) == smiles

        mol = mol_to_standardised_mol(mol)
        assert rdkit.Chem.MolToSmiles(mol) == "CN=c1ccccn1C"

    def test_default_is_unseeded(self):
        import rdkit.Chem
        from rdkit.Chem import AllChem
        from e3fp.conformer.util import (
            mol_from_smiles,
            mol_to_standardised_mol,
        )
        from e3fp.conformer.generate import generate_conformers

        ntrials = 10
        confgen_params = {"num_conf": 1}
        smiles = "C" * 20  # long flexible molecule
        mol = mol_from_smiles(smiles, "tmp")
        mols = [
            generate_conformers(mol, **confgen_params)[0]
            for i in range(ntrials)
        ]

        fail = True
        for i in range(ntrials):
            for j in range(i + 1, ntrials):
                rms = AllChem.GetBestRMS(mols[i], mols[j])
                if rms > 1e-2:
                    fail = False
                    break
        assert not fail

    def test_seed_produces_same_conformers(self):
        import rdkit.Chem
        from rdkit.Chem import AllChem
        from e3fp.conformer.util import (
            mol_from_smiles,
            mol_to_standardised_mol,
        )
        from e3fp.conformer.generate import generate_conformers

        ntrials = 10
        confgen_params = {"num_conf": 1, "seed": 42}
        smiles = "C" * 20  # long flexible molecule
        mol = mol_from_smiles(smiles, "tmp")
        mols = [
            generate_conformers(mol, **confgen_params)[0]
            for i in range(ntrials)
        ]

        fail = False
        for i in range(ntrials):
            for j in range(i + 1, ntrials):
                rms = AllChem.GetBestRMS(mols[i], mols[j])
                if rms > 1e-2:
                    fail = True
                    break
        assert not fail

    def test_compressed_sdf_reads_same_as_uncompressed(self):
        from rdkit import Chem
        from e3fp.conformer.util import mol_from_sdf

        sdf_files = [SDF_FILE_COMPRESSED, SDF_FILE_UNCOMPRESSED]
        smiles = [Chem.MolToSmiles(mol_from_sdf(f)) for f in sdf_files]
        assert smiles[0] == smiles[1]

    def test_conformer_generation_without_name(self):
        from e3fp.conformer.util import mol_from_smiles
        from e3fp.conformer.generate import generate_conformers

        confgen_params = {"num_conf": 1, "seed": 42}
        smiles = "C" * 20  # long flexible molecule
        mol = mol_from_smiles(smiles, "tmp")
        mol.ClearProp("_Name")
        assert not mol.HasProp("_Name")
        generate_conformers(mol, **confgen_params)

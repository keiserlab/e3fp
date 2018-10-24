"""Tests for conformer generation.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import unittest


class ConformerTestCases(unittest.TestCase):
    def test_standardisation(self):
        import rdkit.Chem
        from e3fp.conformer.util import (
            mol_from_smiles,
            mol_to_standardised_mol,
        )

        smiles = "C[N-]c1cccc[n+]1C"
        mol = mol_from_smiles(smiles, "tmp")
        self.assertEqual(rdkit.Chem.MolToSmiles(mol), smiles)

        mol = mol_to_standardised_mol(mol)
        self.assertEqual(rdkit.Chem.MolToSmiles(mol), "CN=c1ccccn1C")


if __name__ == "__main__":
    unittest.main()

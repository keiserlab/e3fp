"""Tests for Shell and Substruct objects.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import pytest

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PLANAR_SDF_FILE = os.path.join(DATA_DIR, "caffeine_planar.sdf.bz2")


class TestShellCreation:
    def test_error_when_center_not_atom(self):
        from e3fp.fingerprint.structs import Shell

        with pytest.raises(TypeError):
            Shell(None)

    def test_error_when_shells_has_non_shell(self):
        from e3fp.fingerprint.structs import Shell

        atom = 0
        shells = [None]
        with pytest.raises(TypeError):
            Shell(atom, shells)

    def test_creation_with_atoms_or_ids_equivalent(self):
        from e3fp.fingerprint.structs import Shell
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        atoms = list(mol.GetAtoms())
        atom_ids = [x.GetIdx() for x in atoms]
        assert Shell(atoms[0], atoms[1:]) == Shell(atom_ids[0], atom_ids[1:])

    def test_create_shell_no_shell(self):
        from e3fp.fingerprint.structs import Shell
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        atoms = list(mol.GetAtoms())
        center_atom = atoms[0]
        Shell(center_atom)

    def test_create_shell_with_same_center_fails(self):
        from e3fp.fingerprint.structs import Shell, FormatError
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        atoms = list(mol.GetAtoms())
        center_atom = atoms[0]
        with pytest.raises(FormatError):
            Shell(center_atom, atoms)

    def test_atoms_converted_to_shells(self):
        from e3fp.fingerprint.structs import Shell
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        atoms = list(mol.GetAtoms())
        center_atom = atoms[0]
        shell = Shell(center_atom, atoms[1:])
        for s in shell.shells:
            assert isinstance(s, Shell)

    def test_creation_with_atoms_or_shells_equal(self):
        from e3fp.fingerprint.structs import Shell
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        atoms = list(mol.GetAtoms())
        shells = list(map(Shell, atoms))
        center_atom = atoms[0]
        shell1 = Shell(center_atom, atoms[1:])
        shell2 = Shell(center_atom, shells[1:])
        assert shell1 == shell2

    def test_recursive_atom_shells_correct(self):
        from e3fp.fingerprint.structs import Shell
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        atoms = list(mol.GetAtoms())
        shell1 = Shell(atoms[5], atoms[6:8])
        shell2 = Shell(atoms[2], atoms[3:5])
        shell = Shell(atoms[0], (shell1, shell2))
        assert shell.atoms == {x.GetIdx() for x in (atoms[0], atoms[2], atoms[5])}


class TestShellComparison:
    def test_shells_same_center_same_atoms_equal(self):
        from e3fp.fingerprint.structs import Shell
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        atoms = list(mol.GetAtoms())
        center_atom = atoms[0]
        shell1 = Shell(center_atom, atoms[1:])
        shell2 = Shell(center_atom, atoms[1:])
        assert shell1 == shell2

    def test_shells_diff_center_same_atoms_nonequal(self):
        from e3fp.fingerprint.structs import Shell
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        atoms = list(mol.GetAtoms())
        shell1 = Shell(atoms[0], atoms[2:])
        shell2 = Shell(atoms[1], atoms[2:])
        assert shell1 != shell2

    def test_shells_same_center_diff_atoms_nonequal(self):
        from e3fp.fingerprint.structs import Shell
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        atoms = list(mol.GetAtoms())
        center_atom = atoms[0]
        shell1 = Shell(center_atom, atoms[1:])
        shell2 = Shell(center_atom, atoms[2:])
        assert shell1 != shell2

    def test_equal_shells_hash_to_same_value(self):
        from e3fp.fingerprint.structs import Shell
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        atoms = list(mol.GetAtoms())
        center_atom = atoms[0]
        shell1 = Shell(center_atom, atoms[1:])
        shell2 = Shell(center_atom, atoms[1:])
        assert hash(shell1) == hash(shell2)

    def test_same_shell_hashes_to_same_value(self):
        from e3fp.fingerprint.structs import Shell
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        atoms = list(mol.GetAtoms())
        center_atom = atoms[0]
        shell = Shell(center_atom, atoms[1:])
        assert hash(shell) == hash(shell)


class TestShellSubstructInterface:
    def test_recursive_shell_substruct_correct1(self):
        from e3fp.fingerprint.structs import Shell
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        atoms = list(mol.GetAtoms())
        shell1 = Shell(atoms[5], atoms[6:8])
        shell2 = Shell(atoms[1], atoms[2:5])
        shell = Shell(atoms[0], (shell1, shell2))
        assert shell.substruct.atoms == {x.GetIdx() for x in atoms[:8]}

    def test_recursive_shell_substruct_correct2(self):
        from e3fp.fingerprint.structs import Shell
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        atoms = list(mol.GetAtoms())
        shell1 = Shell(atoms[1], atoms[2:5])
        shell2 = Shell(atoms[5], {shell1})
        shell3 = Shell(atoms[6], atoms[7:10])
        shell4 = Shell(atoms[10], {shell3})
        shell = Shell(atoms[0], (shell2, shell4))
        assert shell.substruct.atoms == {x.GetIdx() for x in atoms[:11]}

    def test_shell_creation_from_substruct_without_center_fails(self):
        from e3fp.fingerprint.structs import Shell, Substruct, FormatError
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        atoms = list(mol.GetAtoms())
        substruct = Substruct(None, atoms[:2])
        with pytest.raises(FormatError):
            Shell.from_substruct(substruct)

    def test_shell_creation_from_substruct(self):
        from e3fp.fingerprint.structs import Shell, Substruct
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        atoms = list(mol.GetAtoms())
        substruct = Substruct(atoms[0], atoms[:2])
        shell = Shell.from_substruct(substruct)
        assert shell.atoms == substruct.atoms

    def test_substruct_creation_from_shell(self):
        from e3fp.fingerprint.structs import Shell, Substruct
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        atoms = list(mol.GetAtoms())
        shell = Shell(atoms[0], atoms[1:])
        substruct = Substruct.from_shell(shell)
        assert shell.substruct == substruct


class TestSubstructCreation:
    def test_error_when_center_not_atom(self):
        from e3fp.fingerprint.structs import Substruct

        with pytest.raises(TypeError):
            Substruct("foo")

    def test_error_when_atoms_has_non_atom(self):
        from e3fp.fingerprint.structs import Substruct

        atoms = [None]
        with pytest.raises(TypeError):
            Substruct(atoms=atoms)

    def test_center_atom_auto_added_to_atoms(self):
        from e3fp.fingerprint.structs import Substruct
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        atoms = list(mol.GetAtoms())
        center_atom = atoms[0]
        substruct = Substruct(center_atom, atoms[1:])
        assert center_atom.GetIdx() in substruct.atoms


class TestSubstructCreationComparison:
    def test_substructs_same_center_same_atoms_equal(self):
        from e3fp.fingerprint.structs import Substruct
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        atoms = list(mol.GetAtoms())
        center_atom = atoms[0]
        substruct1 = Substruct(center_atom, atoms)
        substruct2 = Substruct(center_atom, atoms)
        assert substruct1 == substruct2

    def test_substructs_diff_center_same_atoms_equal(self):
        from e3fp.fingerprint.structs import Substruct
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        atoms = list(mol.GetAtoms())
        substruct1 = Substruct(atoms[0], atoms)
        substruct2 = Substruct(atoms[1], atoms)
        assert substruct1 == substruct2

    def test_substructs_same_center_diff_atoms_nonequal(self):
        from e3fp.fingerprint.structs import Substruct
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        atoms = list(mol.GetAtoms())
        substruct1 = Substruct(atoms[0], atoms[1:])
        substruct2 = Substruct(atoms[0], atoms[2:])
        assert substruct1 != substruct2

    def test_equal_shells_hash_to_same_value(self):
        from e3fp.fingerprint.structs import Substruct
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        atoms = list(mol.GetAtoms())
        center_atom = atoms[0]
        substruct1 = Substruct(center_atom, atoms[1:])
        substruct2 = Substruct(center_atom, atoms[1:])
        assert hash(substruct1) == hash(substruct2)

    def test_same_shells_hash_to_same_value(self):
        from e3fp.fingerprint.structs import Substruct
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        atoms = list(mol.GetAtoms())
        center_atom = atoms[0]
        substruct = Substruct(center_atom, atoms[1:])
        assert hash(substruct) == hash(substruct)

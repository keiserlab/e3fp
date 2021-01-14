"""Tests for E3FP fingerprinting.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import glob
import os
import pytest
try:
    import mock
except ImportError:
    import unittest.mock as mock

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PLANAR_SDF_FILE = os.path.join(DATA_DIR, "caffeine_planar.sdf.bz2")
NONPLANAR_SDF_FILE = os.path.join(DATA_DIR, "ritalin_nonplanar.sdf.bz2")
RAND_SDF_DIR = os.path.join(DATA_DIR, "rand_sdf_files")
ENANT1_SDF_FILE = os.path.join(DATA_DIR, "stereo1.sdf.bz2")
ENANT2_SDF_FILE = os.path.join(DATA_DIR, "stereo2.sdf.bz2")


class TestShellCreation:
    def test_atom_coords_calculated_correctly(self):
        from e3fp.fingerprint.fprinter import coords_from_atoms
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        conf = mol.GetConformers()[0]
        atoms = [x.GetIdx() for x in mol.GetAtoms()]
        for atom in atoms:
            conf.SetAtomPosition(atom, [0, 0, 0])
        atom_coords = coords_from_atoms(atoms, conf)
        expected_coords = dict(
            list(zip(atoms, np.zeros((len(atoms), 3), dtype=np.float)))
        )
        np.testing.assert_equal(atom_coords, expected_coords)

    def test_distance_matrix_calculated_correctly1(self):
        from e3fp.fingerprint.array_ops import make_distance_matrix

        coords = np.zeros((2, 3), dtype=np.float)
        dist_mat = make_distance_matrix(coords)
        np.testing.assert_almost_equal(
            dist_mat, np.zeros((2, 2), dtype=np.float)
        )

    def test_distance_matrix_calculated_correctly2(self):
        from e3fp.fingerprint.array_ops import make_distance_matrix

        coords = np.array([[0, 0, 0], [-1, -1, -1]], dtype=np.float)
        dist_mat = make_distance_matrix(coords)
        np.testing.assert_almost_equal(
            dist_mat, (3 ** 0.5) * (1 - np.eye(2, dtype=np.float))
        )

    def test_shells_generator_creation_success(self):
        from e3fp.fingerprint.fprinter import ShellsGenerator
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        conf = mol.GetConformers()[0]
        atoms = [x.GetIdx() for x in mol.GetAtoms()]
        ShellsGenerator(
            conf, atoms, radius_multiplier=0.5, include_disconnected=True
        )

    def test_shells_generator_next_works_correctly(self):
        from e3fp.fingerprint.fprinter import ShellsGenerator
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        conf = mol.GetConformers()[0]
        atoms = [x.GetIdx() for x in mol.GetAtoms()]
        shells_gen1 = ShellsGenerator(
            conf, atoms, radius_multiplier=0.5, include_disconnected=True
        )
        shells_gen2 = ShellsGenerator(
            conf, atoms, radius_multiplier=0.5, include_disconnected=True
        )
        assert next(shells_gen1) == next(shells_gen2)

    def test_connected_match_atoms_rad0_correct(self):
        from e3fp.fingerprint.fprinter import ShellsGenerator
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        conf = mol.GetConformers()[0]
        atoms = list(range(3))
        shells_gen = ShellsGenerator(
            conf, atoms, radius_multiplier=0.5, include_disconnected=True
        )
        match_atoms = shells_gen.get_match_atoms(0.0)
        expect_match_atoms = {k: set() for k in atoms}
        assert match_atoms == expect_match_atoms

    def test_connected_match_atoms_rad1_correct1(self):
        from e3fp.fingerprint.fprinter import ShellsGenerator
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        conf = mol.GetConformers()[0]
        atoms = list(range(3))
        for atom in atoms:
            conf.SetAtomPosition(atom, [0, 0, 0])
        shells_gen = ShellsGenerator(
            conf, atoms, radius_multiplier=0.5, include_disconnected=True
        )
        match_atoms = shells_gen.get_match_atoms(1.0)
        expect_match_atoms = {k: (set(atoms) ^ {k}) for k in atoms}
        assert match_atoms == expect_match_atoms

    def test_connected_match_atoms_rad1_correct2(self):
        from e3fp.fingerprint.fprinter import ShellsGenerator
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        conf = mol.GetConformers()[0]
        atoms = list(range(3))
        for atom in atoms:
            conf.SetAtomPosition(atom, [0, 0, atom * 0.75])
        shells_gen = ShellsGenerator(
            conf, atoms, radius_multiplier=0.5, include_disconnected=True
        )
        match_atoms = shells_gen.get_match_atoms(1.0)
        expect_match_atoms = {0: {1}, 1: {0, 2}, 2: {1}}
        assert match_atoms == expect_match_atoms

    def test_generates_correct_connected_shells_level0(self):
        from e3fp.fingerprint.fprinter import ShellsGenerator
        from e3fp.fingerprint.structs import Shell
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        conf = mol.GetConformers()[0]
        atoms = list(range(3))
        expected_shells_dict = {0: Shell(0), 1: Shell(1), 2: Shell(2)}
        shells_gen = ShellsGenerator(conf, atoms, include_disconnected=False)
        shells_dict = next(shells_gen)
        assert shells_dict == expected_shells_dict

    def test_generates_correct_disconnected_shells_level0(self):
        from e3fp.fingerprint.fprinter import ShellsGenerator
        from e3fp.fingerprint.structs import Shell
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        conf = mol.GetConformers()[0]
        atoms = list(range(3))
        expected_shells_dict = {0: Shell(0), 1: Shell(1), 2: Shell(2)}
        shells_gen = ShellsGenerator(conf, atoms)
        shells_dict = next(shells_gen)
        assert shells_dict == expected_shells_dict

    def test_generates_correct_disconnected_shells_level1(self):
        from e3fp.fingerprint.fprinter import ShellsGenerator
        from e3fp.fingerprint.structs import Shell
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        conf = mol.GetConformers()[0]
        atoms = list(range(3))
        for atom in atoms:
            conf.SetAtomPosition(atom, [0, 0, 0.45 * atom])
        expected_shells_dict = {
            0: Shell(0, {1}),
            1: Shell(1, {0, 2}),
            2: Shell(2, {1}),
        }
        shells_gen = ShellsGenerator(
            conf, atoms, radius_multiplier=0.5, include_disconnected=True
        )
        for i in range(2):
            shells_dict = next(shells_gen)
        assert shells_dict == expected_shells_dict

    def test_generates_correct_connected_shells_level1(self):
        from e3fp.fingerprint import fprinter
        from e3fp.fingerprint.structs import Shell
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        conf = mol.GetConformers()[0]
        atoms = list(range(3))
        bonds_dict = {0: {1, 2}, 1: {0}, 2: {0}}
        for atom in atoms:
            conf.SetAtomPosition(atom, [0, 0, 0.45 * atom])
        expected_shells_dict = {
            0: Shell(0, {1}),
            1: Shell(1, {0}),
            2: Shell(2, {}),
        }

        with mock.patch(
            "e3fp.fingerprint.fprinter.bound_atoms_from_mol",
            return_value=bonds_dict,
        ):
            shells_gen = fprinter.ShellsGenerator(
                conf, atoms, radius_multiplier=0.5, include_disconnected=False
            )
            for i in range(2):
                shells_dict = next(shells_gen)
        assert shells_dict == expected_shells_dict

    def test_generates_correct_disconnected_shells_level2(self):
        from e3fp.fingerprint.fprinter import ShellsGenerator
        from e3fp.fingerprint.structs import Shell
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        conf = mol.GetConformers()[0]
        atoms = list(range(3))
        for atom in atoms:
            conf.SetAtomPosition(atom, [0, 0, 0.45 * atom])
        expected_shells_dict1 = {
            0: Shell(0, {1}),
            1: Shell(1, {0, 2}),
            2: Shell(2, {1}),
        }
        expected_shells_dict2 = {
            0: Shell(0, {expected_shells_dict1[1], expected_shells_dict1[2]}),
            1: Shell(1, {expected_shells_dict1[0], expected_shells_dict1[2]}),
            2: Shell(2, {expected_shells_dict1[0], expected_shells_dict1[1]}),
        }
        shells_gen = ShellsGenerator(
            conf, atoms, radius_multiplier=0.5, include_disconnected=True
        )
        for i in range(3):
            shells_dict = next(shells_gen)
        assert shells_dict == expected_shells_dict2

    def test_generates_correct_connected_shells_level2(self):
        from e3fp.fingerprint import fprinter
        from e3fp.fingerprint.structs import Shell
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        conf = mol.GetConformers()[0]
        atoms = list(range(3))
        bonds_dict = {0: {1, 2}, 1: {0}, 2: {0}}
        for atom in atoms:
            conf.SetAtomPosition(atom, [0, 0, 0.45 * atom])
        expected_shells_dict1 = {
            0: Shell(0, {1}),
            1: Shell(1, {0}),
            2: Shell(2, {}),
        }
        expected_shells_dict2 = {
            0: Shell(0, {expected_shells_dict1[1], expected_shells_dict1[2]}),
            1: Shell(1, {expected_shells_dict1[0]}),
            2: Shell(2, {expected_shells_dict1[0]}),
        }
        with mock.patch(
            "e3fp.fingerprint.fprinter.bound_atoms_from_mol",
            return_value=bonds_dict,
        ):
            shells_gen = fprinter.ShellsGenerator(
                conf, atoms, radius_multiplier=0.5, include_disconnected=False
            )
            for i in range(3):
                shells_dict = next(shells_gen)
            assert shells_dict == expected_shells_dict2

    def test_disconnected_substructs_converge(self):
        from e3fp.fingerprint.fprinter import ShellsGenerator
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        conf = mol.GetConformers()[0]
        atoms = list(range(3))
        for atom in atoms:
            conf.SetAtomPosition(atom, [0, 0, 0.45 * atom])
        shells_gen = ShellsGenerator(
            conf, atoms, radius_multiplier=0.5, include_disconnected=True
        )
        for i in range(4):
            shells_dict = next(shells_gen)
            substructs_dict = {k: v.substruct for k, v in shells_dict.items()}
        next_shells_dict = next(shells_gen)
        next_substructs_dict = {
            k: v.substruct for k, v in next_shells_dict.items()
        }

        assert substructs_dict == next_substructs_dict

    def test_connected_substructs_converge(self):
        from e3fp.fingerprint import fprinter
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        conf = mol.GetConformers()[0]
        atoms = list(range(3))
        bonds_dict = {0: {1, 2}, 1: {0}, 2: {0}}
        for atom in atoms:
            conf.SetAtomPosition(atom, [0, 0, 0.45 * atom])
        with mock.patch(
            "e3fp.fingerprint.fprinter.bound_atoms_from_mol",
            return_value=bonds_dict,
        ):
            shells_gen = fprinter.ShellsGenerator(
                conf, atoms, radius_multiplier=0.5, include_disconnected=False
            )
            for i in range(4):
                shells_dict = next(shells_gen)
                substructs_dict = {
                    k: v.substruct for k, v in shells_dict.items()
                }

            next_shells_dict = next(shells_gen)
            next_substructs_dict = {
                k: v.substruct for k, v in next_shells_dict.items()
            }

            assert substructs_dict == next_substructs_dict


class TestArrayVector:
    def test_rot_matrix_vector_rotation_correct(self):
        from e3fp.fingerprint.array_ops import make_rotation_matrix, as_unit

        for i in range(2, 5):
            u0 = as_unit(np.random.uniform(size=3))
            u1 = as_unit(np.random.uniform(size=3))
            rot = make_rotation_matrix(u0, u1)
            np.testing.assert_array_almost_equal(u1, np.dot(rot, u0))

    def test_rot_matrix_array_rotation_correct(self):
        from e3fp.fingerprint.array_ops import make_rotation_matrix, as_unit

        for i in range(1, 5):
            u0 = as_unit(np.random.uniform(size=3))
            arr = np.random.uniform(size=(i, 3))
            v = arr[0, :]
            rot = make_rotation_matrix(v, u0)
            rot_arr = np.dot(rot, arr.T).T
            u1 = as_unit(rot_arr[0, :])
            np.testing.assert_array_almost_equal(u0.flatten(), u1.flatten())

    def test_transform_matrix_rotation_only_correct(self):
        from e3fp.fingerprint.array_ops import (
            make_transform_matrix,
            as_unit,
            pad_array,
            unpad_array,
            Y_AXIS,
        )

        for i in range(1, 5):
            center = np.zeros(3, dtype=np.float)
            arr = np.random.uniform(size=(i, 3)) + center
            y = arr[0, :]
            trans_mat = make_transform_matrix(center, y)
            pad_arr = pad_array(arr)
            rot_arr = unpad_array(np.dot(trans_mat, pad_arr.T).T)
            y0 = as_unit(rot_arr[0, :])
            np.testing.assert_array_almost_equal(
                y0.flatten(), Y_AXIS.flatten()
            )

    def test_transform_matrix_with_translation_correct(self):
        from e3fp.fingerprint.array_ops import (
            make_transform_matrix,
            as_unit,
            pad_array,
            unpad_array,
            Y_AXIS,
        )

        for i in range(2, 7):
            arr = np.random.uniform(size=(i, 3))
            center = arr[0, :]
            y = arr[1, :] - center
            trans_mat = make_transform_matrix(center, y)
            pad_arr = pad_array(arr)
            rot_arr = unpad_array(np.dot(trans_mat, pad_arr.T).T)
            c0 = rot_arr[0, :]
            y0 = as_unit(rot_arr[1, :])
            np.testing.assert_array_almost_equal(c0.flatten(), np.zeros(3))
            np.testing.assert_array_almost_equal(
                y0.flatten(), Y_AXIS.flatten()
            )

    def test_two_axis_transform_correct1(self):
        from e3fp.fingerprint.array_ops import (
            make_transform_matrix,
            as_unit,
            pad_array,
            unpad_array,
            Y_AXIS,
        )

        for i in range(3, 8):
            arr = np.random.uniform(size=(i, 3))
            center = arr[0, :]
            y = arr[1, :] - center
            z = arr[2, :] - center
            trans_mat = make_transform_matrix(center, y, z)
            pad_arr = pad_array(arr)
            rot_arr = unpad_array(np.dot(trans_mat, pad_arr.T).T)
            c0 = rot_arr[0, :]
            y0 = as_unit(rot_arr[1, :])
            z0 = as_unit(rot_arr[2, :])
            np.testing.assert_array_almost_equal(c0.flatten(), np.zeros(3))
            np.testing.assert_array_almost_equal(
                y0.flatten(), Y_AXIS.flatten()
            )
            assert z0[0] == pytest.approx(0.0)

    def test_two_axis_transform_correct2(self):
        from e3fp.fingerprint.array_ops import (
            make_transform_matrix,
            as_unit,
            transform_array,
            Y_AXIS,
        )

        for i in range(3, 8):
            arr = np.random.uniform(size=(i, 3))
            center = arr[0, :]
            y = arr[1, :] - center
            z = arr[2, :] - center
            trans_mat = make_transform_matrix(center, y, z)
            rot_arr = transform_array(trans_mat, arr)
            c0 = rot_arr[0, :]
            y0 = as_unit(rot_arr[1, :])
            z0 = as_unit(rot_arr[2, :])
            np.testing.assert_array_almost_equal(c0.flatten(), np.zeros(3))
            np.testing.assert_array_almost_equal(
                y0.flatten(), Y_AXIS.flatten()
            )
            assert z0[0] == pytest.approx(0.0)


class TestFingerprinterCreation:
    def test_main_parameter_ranges_run_without_fail(self):
        from e3fp.fingerprint.fprinter import Fingerprinter
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        conf = mol.GetConformers()[0]
        stereo_opts = (True, False)
        counts_opts = (True, False)
        include_disconnected_opts = (True, False)
        for bits in (1024, 4096, 2 ** 32):
            for level, remove_substructs in [(-1, True), (5, False)]:
                for stereo in stereo_opts:
                    for counts in counts_opts:
                        for include_disconnected in include_disconnected_opts:
                            fprinter = Fingerprinter(
                                bits=bits,
                                level=level,
                                stereo=stereo,
                                counts=counts,
                                remove_duplicate_substructs=remove_substructs,
                                include_disconnected=include_disconnected,
                            )
                            fprinter.run(conf, mol)
                            fprinter.get_fingerprint_at_level()

    def test_repeated_runs_produce_same_results(self):
        from e3fp.fingerprint import fprinter
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        level = 2
        conf = mol.GetConformers()[0]
        ref_identifiers = None
        for i in range(5):
            fpr = fprinter.Fingerprinter(
                level=level, stereo=True, radius_multiplier=1.718
            )
            fpr.run(conf, mol)
            identifiers = sorted(
                [x.identifier for x in fpr.level_shells[level]]
            )
            if ref_identifiers is None:
                ref_identifiers = identifiers
            else:
                assert identifiers == ref_identifiers

    def test_fingerprint_is_transform_invariant(self):
        from e3fp.fingerprint import fprinter
        from e3fp.fingerprint.array_ops import (
            make_transform_matrix,
            transform_array,
        )
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        level = 5
        conf = mol.GetConformers()[0]
        ref_fp = None
        atom_ids = [x.GetIdx() for x in mol.GetAtoms()]
        coords = np.array(
            list(map(conf.GetAtomPosition, atom_ids)), dtype=np.float
        )
        for i in range(5):
            rand_y = np.random.uniform(size=3)
            rand_trans = np.random.uniform(size=3) * 100
            trans_mat = make_transform_matrix(rand_trans)
            rot_mat = make_transform_matrix(np.zeros(3), rand_y)
            transform_mat = np.dot(trans_mat, rot_mat)
            new_coords = transform_array(transform_mat, coords)
            with pytest.raises(AssertionError):
                np.testing.assert_almost_equal(new_coords, coords)

            for atom_id, new_coord in zip(atom_ids, new_coords):
                conf.SetAtomPosition(atom_id, new_coord)
            test_coords = np.array(
                list(map(conf.GetAtomPosition, atom_ids)), dtype=np.float
            )
            np.testing.assert_almost_equal(test_coords, new_coords)

            fpr = fprinter.Fingerprinter(
                level=level, stereo=True, radius_multiplier=1.718
            )
            fpr.run(conf, mol)
            fp = fpr.get_fingerprint_at_level(level)
            if ref_fp is None:
                ref_fp = fp
            else:
                assert fp == ref_fp

    def test_quick(self):
        from e3fp.fingerprint import fprinter
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        level = 5
        conf = mol.GetConformers()[0]
        fpr = fprinter.Fingerprinter(
            level=level, bits=1024, stereo=True, radius_multiplier=1.718
        )
        fpr.run(conf, mol)
        # fprint = fpr.get_fingerprint_at_level(level)
        # fpr.substructs_to_pdb(reorient=True)


class TestHashing:
    def test_hashing_wrong_dtype_fails1(self):
        from e3fp.fingerprint.fprinter import hash_int64_array

        with pytest.raises(TypeError):
            hash_int64_array(np.arange(3, dtype=np.float))

    def test_hashing_wrong_dtype_fails2(self):
        from e3fp.fingerprint.fprinter import hash_int64_array

        with pytest.raises(TypeError):
            hash_int64_array(np.arange(3, dtype=np.int32))

    def test_hashing_produces_example_result(self):
        from e3fp.fingerprint.fprinter import hash_int64_array

        assert hash_int64_array(np.array([42], dtype=np.int64)) == 1871679806

    def test_mmh3_produces_example_result(self):
        import mmh3

        assert mmh3.hash("foo", 0) == -156908512


class TestStereo:
    def test_stereo_indicators_for_frame(self):
        from e3fp.fingerprint.fprinter import stereo_indicators_from_shell
        from e3fp.fingerprint.array_ops import (
            project_to_plane,
            make_transform_matrix,
            transform_array,
        )
        from e3fp.fingerprint.structs import Shell

        shell = Shell(0, {1, 2, 3})
        atom_coords = np.asarray(
            [
                [0, 0, 0.0],
                [1, -0.5, 0.0],
                [0, 2.0, 0.0],
                [0, 0, 3.0],
            ],  # -> y  # -> z
            dtype=np.float,
        )
        atom_tuples = [
            (1, 1, Shell(2)),  # -> y
            (2, 1, Shell(1)),
            (5, 5, Shell(3)),
        ]  # -> z
        for i in range(20):
            rand_trans = np.random.uniform(size=3) * 100
            rand_y = np.random.uniform(size=3) * 10
            rand_v = np.random.uniform(size=3) * 20
            rand_z = project_to_plane(rand_v, rand_y) * 30
            rand_transform_mat = make_transform_matrix(
                np.zeros(3), rand_y, rand_z
            )
            new_coords = transform_array(rand_transform_mat, atom_coords)
            np.testing.assert_almost_equal(
                atom_coords,
                transform_array(np.linalg.inv(rand_transform_mat), new_coords),
            )
            new_coords += rand_trans

            atom_coords_dict = dict(list(zip(list(range(4)), new_coords)))
            stereo_ind = stereo_indicators_from_shell(
                shell, atom_tuples, atom_coords_dict
            )
            # 2 is chosen for y, 3 for z
            expect_stereo_ind = [1, -5, 2]
            assert stereo_ind == expect_stereo_ind

    def test_empty_tuples_returns_empty(self):
        from e3fp.fingerprint.fprinter import stereo_indicators_from_shell
        from e3fp.fingerprint.structs import Shell

        shell = Shell(0, {})
        atom_coords_dict = {0: np.random.uniform(size=3)}
        atom_tuples = []
        stereo_ind = stereo_indicators_from_shell(
            shell, atom_tuples, atom_coords_dict
        )
        assert len(stereo_ind) == 0

    def test_no_unique_y_two_evenly_spaced_correct(self):
        from e3fp.fingerprint.fprinter import stereo_indicators_from_shell
        from e3fp.fingerprint.structs import Shell

        shell = Shell(0, {1, 2})
        atom_coords_dict = {0: [0, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}
        atom_tuples = [(1, 1, Shell(1)), (1, 1, Shell(2))]
        stereo_ind = stereo_indicators_from_shell(
            shell, atom_tuples, atom_coords_dict
        )
        # mean should be between 1 and 2, so z cannot be picked, so all 0.
        expect_stereo_ind = [1, 2]

        assert stereo_ind == expect_stereo_ind

    def test_no_unique_y_three_evenly_spaced_produces_zeros(self):
        from e3fp.fingerprint.fprinter import stereo_indicators_from_shell
        from e3fp.fingerprint.structs import Shell

        shell = Shell(0, {1, 2, 3})
        atom_coords_dict = {
            0: [0, 0, 0],
            1: [0, 1, 0],
            2: [0, 0, 1],
            3: [1, 0, 0],
        }
        atom_tuples = [(1, 1, Shell(1)), (1, 1, Shell(2)), (1, 1, Shell(3))]
        stereo_ind = stereo_indicators_from_shell(
            shell, atom_tuples, atom_coords_dict
        )
        # mean should be between 1 and 2, so z cannot be picked, so all 0.
        expect_stereo_ind = [0, 0, 0]

        assert stereo_ind == expect_stereo_ind

    def test_no_unique_y_along_poles_correct(self):
        from e3fp.fingerprint.fprinter import stereo_indicators_from_shell
        from e3fp.fingerprint.structs import Shell

        shell = Shell(0, {1, 2, 3})
        atom_coords_dict = {
            0: [0, 0, 0],
            1: [0, 1, 0],
            2: [0, -5, 0],
            3: [0, -10, 0],
        }
        atom_tuples = [(1, 1, Shell(1)), (1, 1, Shell(2)), (1, 1, Shell(3))]
        stereo_ind = stereo_indicators_from_shell(
            shell, atom_tuples, atom_coords_dict
        )
        # mean should be along atom 2/3, so z not be picked, but all at poles.
        expect_stereo_ind = [-1, 1, 1]

        assert stereo_ind == expect_stereo_ind

    def test_all_indicators_correctly_assigned(self):
        from e3fp.fingerprint.fprinter import quad_indicators_from_coords

        cent_coords = np.array(
            [
                [0.0, 1.0, 1.0],
                [-1.0, 1.0, 0.0],
                [0.0, 1.0, -1.0],
                [1.0, 1.0, 0.0],
                [0.0, -1.0, 1.0],
                [-1.0, -1.0, 0.0],
                [0.0, -1.0, -1.0],
                [1.0, -1.0, 0.0],
            ],
            dtype=np.float,
        )
        y = np.array([0, 1, 0], dtype=np.float)
        y_ind = None
        z = np.array([0, 0, 1], dtype=np.float)
        long_sign = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.int64)
        quad = quad_indicators_from_coords(cent_coords, y, y_ind, z, long_sign)
        expect_quad = [2, 3, 4, 5, -2, -3, -4, -5]
        assert list(quad) == expect_quad

    def test_stereo_sets_correct_transform_matrix(self):
        from e3fp.fingerprint.fprinter import stereo_indicators_from_shell
        from e3fp.fingerprint.array_ops import (
            project_to_plane,
            make_transform_matrix,
            transform_array,
        )
        from e3fp.fingerprint.structs import Shell

        shell = Shell(0, {1, 2})
        atom_coords = np.asarray(
            [[0, 0, 0.0], [0, 2.0, 0.0], [0, 0, 3.0]],
            dtype=np.float,  # -> y  # -> z
        )
        atom_tuples = [(1, 1, Shell(1)), (5, 5, Shell(2))]  # -> y  # -> z
        for i in range(20):
            rand_trans = np.random.uniform(size=3) * 100
            rand_y = np.random.uniform(size=3) * 10
            rand_v = np.random.uniform(size=3) * 20
            rand_z = project_to_plane(rand_v, rand_y) * 30
            rand_transform_mat = make_transform_matrix(
                np.zeros(3), rand_y, rand_z
            )
            trans_mat = np.identity(4, dtype=np.float)
            trans_mat[:3, 3] = rand_trans
            rand_transform_mat = np.dot(trans_mat, rand_transform_mat)

            new_coords = transform_array(rand_transform_mat, atom_coords)
            reverse_trans_mat = np.linalg.inv(rand_transform_mat)

            np.testing.assert_almost_equal(
                atom_coords, transform_array(reverse_trans_mat, new_coords)
            )

            atom_coords_dict = dict(list(zip(list(range(3)), new_coords)))
            stereo_indicators_from_shell(shell, atom_tuples, atom_coords_dict)
            np.testing.assert_almost_equal(
                shell.transform_matrix, reverse_trans_mat
            )

    def test_no_neighbors_transform_matrix_is_translate(self):
        from e3fp.fingerprint.fprinter import stereo_indicators_from_shell
        from e3fp.fingerprint.structs import Shell

        shell = Shell(0)
        center_coord = np.random.uniform(3)
        stereo_indicators_from_shell(shell, [], {0: center_coord})
        np.testing.assert_almost_equal(
            shell.transform_matrix[:3, 3], -center_coord
        )

    # def test_stereo_planar_molecule_successful(self):
    #     pass

    # def test_more_uniqueness_usecases(self):
    #     pass


class TestGenerateFingerprint:
    def test_runs_without_exception_on_random_mols(self):
        from e3fp.fingerprint import fprinter
        from e3fp.conformer.util import mol_from_sdf

        rand_sdf_files = glob.glob(RAND_SDF_DIR + "/*.sdf*")
        rand_sdf_files = [
            rand_sdf_files[i]
            for i in np.random.randint(len(rand_sdf_files), size=10)
        ]
        level = 5
        for sdf_file in rand_sdf_files:
            mol = mol_from_sdf(sdf_file)
            conf = mol.GetConformers()[0]
            fpr = fprinter.Fingerprinter(
                level=level, bits=1024, stereo=True, radius_multiplier=1.718
            )
            fpr.run(conf, mol)
            fpr.get_fingerprint_at_level(level)

    def test_initial_identifiers_assigned_correctly(self):
        from e3fp.fingerprint import fprinter
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        level = 0
        conf = mol.GetConformers()[0]
        fpr = fprinter.Fingerprinter(
            level=level, bits=1024, stereo=True, radius_multiplier=1.718
        )
        fpr.run(conf, mol)
        fprint = fpr.get_fingerprint_at_level(0)
        expect_ident = set([48, 124, 185, 484, 617, 674])
        assert set(fprint.indices) == expect_ident

    def test_remove_dupe_substructs_makes_same_substruts_diff_shells(self):
        from e3fp.fingerprint import fprinter
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        level = 2
        conf = mol.GetConformers()[0]
        fpr = fprinter.Fingerprinter(
            level=level,
            bits=1024,
            stereo=True,
            radius_multiplier=1.718,
            remove_duplicate_substructs=True,
        )
        fpr.run(conf, mol)
        shells_no_dupes = set(fpr.level_shells[fpr.current_level])
        substructs_no_dupes = set([x.substruct for x in shells_no_dupes])

        fpr = fprinter.Fingerprinter(
            level=level,
            bits=1024,
            stereo=True,
            radius_multiplier=1.718,
            remove_duplicate_substructs=False,
        )
        fpr.run(conf, mol)
        shells_with_dupes = set(fpr.level_shells[fpr.current_level])
        substructs_with_dupes = set([x.substruct for x in shells_with_dupes])

        assert substructs_no_dupes == substructs_with_dupes
        assert shells_no_dupes != shells_with_dupes

    def test_stereoisomers_produce_nonequal_fingerprints_stereo(self):
        from e3fp.fingerprint import fprinter
        from e3fp.conformer.util import mol_from_sdf

        mol1 = mol_from_sdf(ENANT1_SDF_FILE)
        mol2 = mol_from_sdf(ENANT2_SDF_FILE)
        level = 5
        fpr = fprinter.Fingerprinter(
            level=level,
            bits=1024,
            stereo=True,
            radius_multiplier=1.718,
            remove_duplicate_substructs=True,
        )
        fpr.run(conf=0, mol=mol1)
        fp1 = fpr.get_fingerprint_at_level(level)
        fpr.run(conf=0, mol=mol2)
        fp2 = fpr.get_fingerprint_at_level(level)
        assert fp1 != fp2

    def test_stereoisomers_produce_equal_fingerprints_nonstereo(self):
        from e3fp.fingerprint import fprinter
        from e3fp.conformer.util import mol_from_sdf

        mol1 = mol_from_sdf(ENANT1_SDF_FILE)
        mol2 = mol_from_sdf(ENANT2_SDF_FILE)
        level = 5
        fpr = fprinter.Fingerprinter(
            level=level,
            stereo=False,
            radius_multiplier=1.718,
            remove_duplicate_substructs=True,
        )
        fpr.run(conf=0, mol=mol1)
        fp1 = fpr.get_fingerprint_at_level(level)
        fpr.run(conf=0, mol=mol2)
        fp2 = fpr.get_fingerprint_at_level(level)
        assert fp1 == fp2

    def test_reordering_conformers_produces_same_fprints(self):
        from e3fp.fingerprint import fprinter
        from e3fp.conformer.util import mol_from_sdf
        import random

        rand_sdf_files = glob.glob(os.path.join(RAND_SDF_DIR, "*.sdf*"))
        mol = mol_from_sdf(rand_sdf_files[0])
        level = 5
        fpr = fprinter.Fingerprinter(
            level=level,
            stereo=False,
            radius_multiplier=1.718,
            remove_duplicate_substructs=True,
        )
        conf_ids1 = [x.GetId() for x in mol.GetConformers()]
        fprints1 = {}
        for conf_id in conf_ids1:
            fpr.run(conf_id, mol)
            fprints1[conf_id] = fpr.get_fingerprint_at_level(level)

        conf_ids2 = list(conf_ids1)
        random.shuffle(conf_ids2)
        fprints2 = {}
        for conf_id in conf_ids2:
            fpr.run(conf_id, mol)
            fprints2[conf_id] = fpr.get_fingerprint_at_level(level)
        assert fprints1 == fprints2

    def test_reordering_mols_produces_same_fprints(self):
        from e3fp.fingerprint import fprinter
        from e3fp.conformer.util import mol_from_sdf
        import random

        rand_sdf_files = glob.glob(RAND_SDF_DIR + "/*.sdf*")
        mols = list(map(mol_from_sdf, rand_sdf_files[:5]))
        level = 5
        fpr = fprinter.Fingerprinter(
            level=level,
            stereo=False,
            radius_multiplier=1.718,
            remove_duplicate_substructs=True,
        )
        fprints1 = {}
        for mol in mols:
            fpr.run(conf=0, mol=mol)
            fprints1[mol] = fpr.get_fingerprint_at_level(level)

        random.shuffle(mols)
        fpr = fprinter.Fingerprinter(
            level=level,
            stereo=False,
            radius_multiplier=1.718,
            remove_duplicate_substructs=True,
        )
        fprints2 = {}
        for mol in mols:
            fpr.run(conf=0, mol=mol)
            fprints2[mol] = fpr.get_fingerprint_at_level(level)

        assert fprints1 == fprints2


class TestAtomInvariant:
    def test_daylight_invariants(self):
        from e3fp.fingerprint import fprinter
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        atom = mol.GetAtomWithIdx(2)
        invars = fprinter.invariants_from_atom(atom)
        assert list(invars) == [2, 3, 6, 12, 0, 1, 1]

    def test_rdkit_invariants(self):
        from e3fp.fingerprint import fprinter
        from e3fp.conformer.util import mol_from_sdf

        mol = mol_from_sdf(PLANAR_SDF_FILE)
        atom = mol.GetAtomWithIdx(2)
        invars = fprinter.rdkit_invariants_from_atom(atom)
        assert list(invars) == [6, 3, 1, 0, 0, 1]

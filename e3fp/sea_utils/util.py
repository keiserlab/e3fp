"""Utilities for input/output of SEA-compatible files and formatting of dicts.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import sys
import logging
import csv
import ast
from collections import OrderedDict

from seacore.util.library import SetValue
from seashell.cli.fputil import FingerprintType
from seashell.common.read_csv import read_csv_mols
from seashell.cli.library_cli import read_targets, write_targets, \
                                        CSV_SUBSEP_TARGETS

from python_utilities.io_tools import smart_open
from e3fp.conformer.util import MolItemName

csv.field_size_limit(sys.maxsize)


def fprint_params_to_fptype(bits=None, radius_multiplier=None,
                            level=None, stereo=None,
                            include_disconnected=None, **kwargs):
    """Make SEA ``FingerprintType`` from fingerprint parameters."""
    if len(kwargs) > 0:
        logging.warning(
            "Ignoring fingerprint params {!r} for fp_type".format(kwargs))
    fp_type = FingerprintType()
    fp_type.data[fp_type.KEYS[0]] = 'e3fp'
    fp_type.data[fp_type.KEYS[1]] = 'sea_native'
    fp_type.data[fp_type.KEYS[2]] = bits
    fp_params = {'radius_multiplier': "%.4g" % radius_multiplier,
                 'level': str(level), 'stereo': str(stereo),
                 'include_disconnected': str(include_disconnected)}
    fp_type.data[fp_type.KEYS[3]] = str(sorted(fp_params.items()))
    return fp_type


def fptype_to_fprint_params(fp_type):
    """From SEA ``FingerprintType``, make fingerprint parameters dict."""
    fp_params = {}
    fp_params["bits"] = int(fp_type.data[fp_type.KEYS[2]])
    _fp_params = dict(ast.literal_eval(fp_type.data[fp_type.KEYS[3]]))
    if "radius_multiplier" in _fp_params:
        fp_params["radius_multiplier"] = float(
            _fp_params["radius_multiplier"])
    if "level" in _fp_params:
        fp_params["level"] = int(_fp_params["level"])
    if "stereo" in _fp_params:
        fp_params["stereo"] = ast.literal_eval(_fp_params["stereo"])
    if "include_disconnected" in _fp_params:
        fp_params["include_disconnected"] = ast.literal_eval(
            _fp_params["include_disconnected"])
    return fp_params


def targets_to_mol_lists_targets(targets_dict, mol_lists_dict):
    """Convert targets with only mol names to targets with mol lists."""
    mol_lists_targets_dict = {}
    for target_key, set_value in targets_dict.iteritems():
        mol_lists_cids = [
            CSV_SUBSEP_TARGETS.join([x[1] for x in mol_lists_dict[cid]])
            for cid in set_value.cids if cid in mol_lists_dict]
        if len(mol_lists_cids) == 0:
            continue
        mol_lists_set_value = SetValue(set_value.name, mol_lists_cids,
                                       set_value.description)
        mol_lists_targets_dict[target_key] = mol_lists_set_value
    return mol_lists_targets_dict


def mol_lists_targets_to_targets(mol_lists_targets_dict, merge_proto=True):
    """Convert targets with mol lists to targets with only mol mol_names."""
    targets_dict = {}
    for target_key, mol_lists_set_value in mol_lists_targets_dict.iteritems():
        if merge_proto:
            cids_dict = OrderedDict([(MolItemName.from_str(x).mol_name, None)
                                     for x in mol_lists_set_value.cids])
        else:
            cids_dict = OrderedDict([(MolItemName.from_str(x).proto_name, None)
                                     for x in mol_lists_set_value.cids])
        set_value = SetValue(mol_lists_set_value.name, cids_dict.keys(),
                             mol_lists_set_value.description)
        targets_dict[target_key] = set_value
    return targets_dict


def targets_to_dict(targets_file, affinity=None):
    """Read targets file to a dict. Optionally filter by affinity."""
    targets_dict = read_targets(targets_file)
    if affinity is not None:
        filtered_targets_dict = dict([(k, v) for k, v
                                      in targets_dict.iteritems()
                                      if str(k.group) == str(affinity)])
        targets_dict = filtered_targets_dict
    return targets_dict


def dict_to_targets(targets_file, targets_dict):
    """Write targets dict to file."""
    write_targets(targets_file, targets_dict)


def molecules_to_lists_dicts(molecules_file, first=-1, merge_proto=True):
    """Read molecules file to dict of mol names to list of native tuples."""
    proto_smiles_dict = {}
    proto_lists_dict = {}
    smiles_dict = {}
    mol_lists_dict = {}
    mol_iter = read_csv_mols(molecules_file, has_fp=True)
    fp_type = mol_iter.next()
    while True:
        try:
            row = mol_iter.next()
        except StopIteration:
            break
        if row is None:
            continue
        fp_name, smiles = row[:2]

        proto_name = MolItemName.from_str(
            MolItemName.from_str(fp_name).proto_name)
        proto_smiles_dict[proto_name] = smiles
        try:
            fp_native = row[2]
            proto_lists_dict.setdefault(proto_name, []).append((fp_native,
                                                                fp_name))
            # mol_lists_dict.setdefault(mol_name, []).append((fp_native,
            #                                                 fp_name))
        except IndexError:
            logging.warning("%s has no native fingerprint. Will not be added to dict." % fp_name)

    if first > 0:
        proto_lists_dict = dict([(k, v[:first])
                                 for k, v in proto_lists_dict.iteritems()])

    if merge_proto:
        for mol_item_name, native_tuples in sorted(proto_lists_dict.items()):
            mol_name = mol_item_name.mol_name
            mol_lists_dict.setdefault(mol_name, []).extend(native_tuples)
            smiles_dict.setdefault(mol_name,
                                   proto_smiles_dict[mol_item_name])
    else:
        for mol_item_name, native_tuples in proto_lists_dict.iteritems():
            proto_name = mol_item_name.proto_name
            mol_lists_dict[proto_name] = native_tuples
            smiles_dict[proto_name] = proto_smiles_dict[mol_item_name]

    return smiles_dict, mol_lists_dict, fp_type


def lists_dicts_to_molecules(molecules_file, smiles_dict, mol_lists_dict,
                             fp_type):
    """Write dict of mol names to list of native tuples to a molecules file."""
    with smart_open(molecules_file, "wb") as f:
        writer = csv.writer(f)
        fp_type.write(writer)
        writer.writerow(("molecule id", "smiles", "fingerprint"))
        for mol_name in sorted(smiles_dict.keys()):
            smiles = smiles_dict[mol_name]
            fp_list = mol_lists_dict.get(mol_name, [])
            for fp_native, fp_name in fp_list:
                writer.writerow((fp_name, smiles, fp_native))


def native_tuples_to_molecules(molecules_file, native_tuples_lists_iter,
                               smiles_dict, fp_type):
    """Given an iterable of native tuples lists, write to molecules file."""
    with smart_open(molecules_file, "wb") as f:
        writer = csv.writer(f)
        fp_type.write(writer)
        writer.writerow(("molecule id", "smiles", "fingerprint"))
        for i, native_tuples_list in enumerate(native_tuples_lists_iter):
            logging.debug(
                "Wrote native strings for molecule {:d} to molecules file.".format(i + 1))
            # smiles = smiles_dict[mol_name]
            for fp_native, fp_name in native_tuples_list:
                mol_item_name = MolItemName.from_str(native_tuples_list[0][1])
                smiles = smiles_dict.get(mol_item_name.proto_name,
                                         smiles_dict.get(
                                            mol_item_name.mol_name))
                writer.writerow((fp_name, smiles, fp_native))


def filter_molecules_by_targets(mol_lists_dict, targets_dict):
    """Return only molecules matching to a target in `targets_dict`."""
    all_mols_set = set().union(*[x.cids for x in targets_dict.itervalues()])
    return dict([(k, v) for k, v in mol_lists_dict.iteritems()
                 if k in all_mols_set])


def filter_targets_by_molecules(targets_dict, mol_lists_dict):
    """Return targets dict including only molecules in `mol_lists_dict`."""
    filtered_targets_dict = {}
    for target_key, set_value in targets_dict.iteritems():
        cids = [x for x in set_value.cids if x in mol_lists_dict]
        if len(cids) == 0:
            continue
        filtered_targets_dict[target_key] = SetValue(set_value.name, cids,
                                                     set_value.description)
    return filtered_targets_dict

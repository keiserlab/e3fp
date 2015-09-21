"""Functions for determining protonation states of molecules.

Author: Seth Axen
E-mail: seth.axen@gmail.com"""
import os
import tempfile
import subprocess
import itertools
import logging

from e3fp.conformer.util import iter_to_smiles, smiles_to_dict

MOL_PROTO_DELIM = ":"


def smiles_to_proto_smiles(out_smiles_file, in_smiles_file, max_states=3,
                           pka=7.4, dist_cutoff=20., parallelizer=None):
    """Convert SMILES file to protonated SMILES file."""
    smiles_dict = smiles_to_dict(in_smiles_file)
    kwargs = {"max_states": max_states, "pka": pka, "dist_cutoff": dist_cutoff}
    if parallelizer is None:
        proto_smiles_iter = iter(smiles_list_to_proto_smiles_list(
            smiles_dict.items(), **kwargs))
    else:
        smiles_chunks_iter = (
            (chunk, ) for chunk in chunk_iter_to_lists(smiles_dict.items(),
                                                       chunk_size=100))
        results_iter = (
            result for result, data in parallelizer.run_gen(
                smiles_list_to_proto_smiles_list, smiles_chunks_iter,
                kwargs=kwargs) if result is not False)
        proto_smiles_iter = itertools.chain.from_iterable(results_iter)

    iter_to_smiles(out_smiles_file, proto_smiles_iter)


def smiles_list_to_proto_smiles_list(in_smiles_list, max_states=3, pka=7.4,
                                     dist_cutoff=20.):
    """Convert list of SMILES (mol_name, smiles) to protonated SMILES list."""
    in_smiles_file = tempfile.mkstemp(suffix=".smi")[1]
    iter_to_smiles(in_smiles_file, iter(in_smiles_list))
    logging.debug("Protonating SMILES in %s" % (in_smiles_file))
    proc = subprocess.Popen(
        "cxcalc %s dominanttautomerdistribution -H %g -C false -t dist | molconvert smiles -T name:dist" % (
            in_smiles_file, pka), shell=True,
        stdout=subprocess.PIPE)

    proto_smiles_list = []
    try:
        stdout_iter = iter(proc.stdout.readline, b'')
        next(stdout_iter)
        curr_mol_name = None
        curr_states_count = 0
        for line in stdout_iter:
            smiles, mol_name, dist = line.rstrip('\r\n').split()
            if mol_name != curr_mol_name:
                curr_states_count = 0
            if curr_states_count >= max_states:
                continue
            if int(dist) > dist_cutoff:
                proto_name = _proto_name_to_mol_name(mol_name,
                                                     curr_states_count)
                curr_states_count += 1
                proto_smiles_list.append((proto_name, smiles))
        logging.debug("Finished protonating SMILES in %s" % (in_smiles_file))
    except:
        logging.exception("Error running cxcalc", exc_info=True)

    proc.kill()
    os.remove(in_smiles_file)
    return proto_smiles_list


def chunk_iter_to_lists(iterable, chunk_size=100):
    """Yield chunks of size `chunk_size` from iterator."""
    i = 0
    chunk = []
    for item in iterable:
        if i >= chunk_size:
            yield chunk
            chunk = []
            i = 0
        chunk.append(item)
        i += 1
    if len(chunk) != 0:
        yield chunk


def _proto_name_to_mol_name(mol_item_name, delim=MOL_PROTO_DELIM):
    """Convert protonation state name to molecule name."""
    return mol_item_name.rsplit(delim)[0]


def _mol_name_to_proto_name(mol_name, state_num, delim=MOL_PROTO_DELIM):
    """Convert molecule name to protonation state name."""
    return "".join(map(str, [mol_name, delim, state_num]))

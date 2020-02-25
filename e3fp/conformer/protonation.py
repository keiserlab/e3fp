"""Functions for generating protonation states of molecules.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import tempfile
import subprocess
import itertools
import logging

from .util import iter_to_smiles, MolItemName


def smiles_dict_to_proto_smiles_dict(
    in_smiles_dict,
    max_states=3,
    pka=7.4,
    dist_cutoff=20.0,
    add_missing=False,
    parallelizer=None,
    chunk_size=100,
):
    """Generate dict of SMILES for protonated states from SMILES dict."""
    kwargs = {"max_states": max_states, "pka": pka, "dist_cutoff": dist_cutoff}
    in_smiles_iter = (
        (smiles, mol_name) for mol_name, smiles in in_smiles_dict.items()
    )
    if parallelizer is None:
        proto_smiles_iter = iter(
            smiles_list_to_proto_smiles_list(in_smiles_iter, **kwargs)
        )
    else:
        smiles_chunks_iter = (
            (chunk,)
            for chunk in _chunk_iter_to_lists(
                in_smiles_iter, chunk_size=chunk_size
            )
        )
        results_iter = (
            result
            for result, data in parallelizer.run_gen(
                smiles_list_to_proto_smiles_list,
                smiles_chunks_iter,
                kwargs=kwargs,
            )
            if result is not False
        )
        proto_smiles_iter = itertools.chain.from_iterable(results_iter)

    proto_smiles_dict = {
        mol_name: smiles for smiles, mol_name in proto_smiles_iter
    }
    if add_missing:
        for mol_name, smiles in in_smiles_dict.items():
            proto_name = MolItemName(mol_name, proto_state_num=0).proto_name
            if proto_name not in proto_smiles_dict:
                logging.debug(
                    (
                        "Protonated SMILES for {} could not be generated. "
                        "Returning input SMILES."
                    ).format(mol_name)
                )
                proto_smiles_dict[mol_name] = smiles

    return proto_smiles_dict


def smiles_list_to_proto_smiles_list(
    in_smiles_list, max_states=3, pka=7.4, dist_cutoff=20.0
):
    """Generate list of SMILES for protonated states from single SMILES."""
    in_smiles_file = tempfile.mkstemp(suffix=".smi")[1]
    iter_to_smiles(
        in_smiles_file,
        ((mol_name, smiles) for smiles, mol_name in in_smiles_list),
    )
    logging.debug("Protonating SMILES in %s" % (in_smiles_file))
    proc = subprocess.Popen(
        (
            "cxcalc %s --ignore-error dominanttautomerdistribution -H %g -C "
            'false -t dist -f "smiles:n,T:dist"'
        ).format(in_smiles_file, pka),
        shell=True,
        stdout=subprocess.PIPE,
    )

    proto_smiles_list = []
    try:
        stdout_iter = iter(proc.stdout.readline, b"")
        next(stdout_iter)
        curr_mol_name = None
        curr_states_count = 0
        for line in stdout_iter:
            try:
                smiles, mol_name, dist = line.rstrip("\r\n").split()
            except ValueError:
                logging.warning("Error parsing line:\n%s" % line)
                continue
            if mol_name != curr_mol_name:
                curr_states_count = 0
                curr_mol_name = mol_name
            if curr_states_count >= max_states:
                continue
            if float(dist) > dist_cutoff:
                proto_name = MolItemName(
                    mol_name, proto_state_num=curr_states_count
                ).proto_name
                curr_states_count += 1
                proto_smiles_list.append((smiles, proto_name))
        logging.debug("Finished protonating SMILES in %s" % (in_smiles_file))
    except Exception:
        logging.exception("Error running cxcalc", exc_info=True)

    proc.kill()
    os.remove(in_smiles_file)
    return proto_smiles_list


def smiles_to_proto_smiles(
    smiles, mol_name, max_states=3, pka=7.4, dist_cutoff=20.0
):
    """Generate list of SMILES for protonated states from single SMILES.

    This is very inefficient in batch.
    """
    logging.debug("Protonating SMILES in %s" % (mol_name))
    proc = subprocess.Popen(
        (
            'cxcalc "%s %s" --ignore-error dominanttautomerdistribution -H %g '
            '-C false -t dist -f "smiles:n,T:dist"'
        ).format(smiles, mol_name, pka),
        shell=True,
        stdout=subprocess.PIPE,
    )
    states_count = 0
    proto_smiles_list = []
    try:
        stdout_iter = iter(proc.stdout.readline, b"")
        next(stdout_iter)
        for line in stdout_iter:
            try:
                this_smiles, this_name, dist = line.rstrip("\r\n").split()
            except ValueError:
                logging.warning("Error parsing line:\n%s" % line)
                continue
            if states_count >= max_states:
                break
            if float(dist) > dist_cutoff:
                proto_name = MolItemName(
                    mol_name, proto_state_num=states_count
                ).proto_name
                states_count += 1
                proto_smiles_list.append((smiles, proto_name))
        logging.debug("Finished protonating SMILES in %s" % (mol_name))
    except OSError:
        logging.exception(
            "Error running cxcalc on %s" % (mol_name), exc_info=True
        )

    proc.kill()
    return proto_smiles_list


def _chunk_iter_to_lists(iterable, chunk_size=100):
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

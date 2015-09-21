"""Generate fingerprints from smiles and save to SEA molecules file.

Author: Seth Axen
E-mail: seth.axen@gmail.com"""
from python_utilities.parallel import Parallelizer
from python_utilities.scripting import setup_logging
from e3fp.conformer.util import smiles_to_dict
from e3fp.sea_utils.util import lists_dicts_to_molecules,\
                                 fprint_params_to_fptype
from e3fp.pipeline import native_tuples_from_smiles

SMILES_FILE = "data/chembl17_smiles.smi"

# set conformer generation, fingerprinting, and folding parameters
confgen_kwargs = {"pool_multiplier": 2, "rmsd_cutoff": 0.5}
fprint_kwargs = {"shell_radius": 1.671, "stereo": True,
                 "include_disconnected": True}
fold_kwargs = {"bits": 1024}
kwargs = {"save": False, "first": 8, "level": 5,
          "confgen_kwargs": confgen_kwargs, "fprint_kwargs": fprint_kwargs,
          "fold_kwargs": fold_kwargs}

# setup
setup_logging()
parallelizer = Parallelizer(parallel_mode="processes")

# read in space-delimited smiles, name from file
smiles_dict = smiles_to_dict(SMILES_FILE)
smiles_iter = ((smiles, name) for name, smiles in smiles_dict.iteritems())

# fingerprint in parallel
mol_list_dict = {data[1]: result for result, data in parallelizer.run_gen(
    native_tuples_from_smiles, smiles_iter, kwargs=kwargs)}

# save to SEA molecules file
fp_type = fprint_params_to_fptype(fold_level=fold_kwargs["bits"],
                                  level=kwargs["level"], **fprint_kwargs)
lists_dicts_to_molecules("chembl17_binding_e3fp_molecules.csv.gz", smiles_dict,
                         mol_list_dict, fp_type)

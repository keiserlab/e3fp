"""Generate fingerprints from smiles and save to SEA molecules file.

Author: Seth Axen
E-mail: seth.axen@gmail.com"""
import logging

from python_utilities.parallel import Parallelizer
from python_utilities.scripting import setup_logging
from e3fp.conformer.util import smiles_to_dict
from e3fp.sea_utils.util import lists_dicts_to_molecules,\
                                 fprint_params_to_fptype
from e3fp.pipeline import native_tuples_from_smiles, params_to_dicts

PARAMS_FILE = "data/params.cfg"
SMILES_FILE = "data/chembl17_smiles.smi"
OUT_FILE = "chembl17_binding_e3fp_molecules.csv.gz"

# set conformer generation, fingerprinting, and folding parameters
confgen_params, fprint_params, fold_params = params_to_dicts(PARAMS_FILE)
kwargs = {"save": False, "confgen_params": confgen_params,
          "fprint_params": fprint_params, "fold_params": fold_params}

# setup
setup_logging()
parallelizer = Parallelizer(parallel_mode="processes")

# read in space-delimited smiles, name from file
logging.info("Reading SMILES from {}".format(SMILES_FILE))
smiles_dict = smiles_to_dict(SMILES_FILE)
smiles_iter = ((smiles, name) for name, smiles in smiles_dict.iteritems())

# fingerprint in parallel
logging.info("Fingerprinting {:d} molecules".format(len(smiles_dict)))
mol_list_dict = {data[1]: result for result, data in parallelizer.run_gen(
    native_tuples_from_smiles, smiles_iter, kwargs=kwargs)}
logging.info("Finished fingerprinting molecules")


# save to SEA molecules file
logging.info("Saving fingerprints to {}".format(OUT_FILE))
fp_type = fprint_params_to_fptype(**fprint_params)
lists_dicts_to_molecules(OUT_FILE, smiles_dict, mol_list_dict, fp_type)
logging.info("Finished!")

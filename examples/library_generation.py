"""Generate SEA library from SEA molecules file.

Author: Seth Axen
E-mail: seth.axen@gmail.com"""
from python_utilities.scripting import setup_logging
from e3fp.sea_utils.util import molecules_to_lists_dicts, dict_to_targets,\
                                targets_to_dict, targets_to_mol_lists_targets
from e3fp.sea_utils.library import build_library

E3FP_MOLECULES_FILE = "chembl17_binding_e3fp_molecules.csv.gz"
ECFP_TARGETS_FILE = "data/chembl17_binding_ecfp4_targets.csv.gz"

# setup
setup_logging()

# load files
smiles_dict, mol_lists_dict, fp_type = molecules_to_lists_dicts(
    E3FP_MOLECULES_FILE)
ecfp_targets_dict = targets_to_dict(ECFP_TARGETS_FILE)

# convert ecfp targets to e3fp targets and save
e3fp_targets_dict = targets_to_mol_lists_targets(ecfp_targets_dict,
                                                 mol_lists_dict)
e3fp_targets_file = "chembl17_binding_e3fp_targets.csv.gz"
dict_to_targets(e3fp_targets_file, e3fp_targets_dict)

# build library and generate fit
build_library("chembl17_e3fp_library.sea", E3FP_MOLECULES_FILE,
              e3fp_targets_file, "chembl17_e3fp_library.fit",
              generate_fit=True)

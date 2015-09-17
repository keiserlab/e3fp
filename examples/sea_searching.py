"""Search molecules against library using SEA.

Author: Seth Axen
E-mail: seth.axen@gmail.com"""
from python_utilities.scripting import setup_logging
from e3fp.sea_utils.util import molecules_to_lists_dicts
from e3fp.sea_utils.run import sea_set_search

QUERY_MOLECULES_FILE = "data/test_e3fp_molecules.csv.gz"
E3FP_LIBRARY_FILE = "chembl17_e3fp_library.sea"

# setup
setup_logging()

# load files
_, mol_lists_dict, _ = molecules_to_lists_dicts(QUERY_MOLECULES_FILE)

# search molecules against library using SEA
searcher = sea_set_search(E3FP_LIBRARY_FILE, mol_lists_dict)

# print output
print("\t".join(["mol_name", "tid", "group", "evalue", "tc"]))
for mol_name in mol_lists_dict:
    results = searcher.mol_result(mol_name)
    for target_key, result_tuple in results.iteritems():
        print("\t".join([mol_name, target_key.tid, target_key.group,
                         "%.4g" % result_tuple[0], "%.4f" % result_tuple[1]]))
searcher.close()

#Extended 3-Dimensional FingerPrint (E3FP)#

E3FP is a 3D molecule fingerprinting method inspired by Extended Connectivity FingerPrints (ECFP).

##Table of Contents##
- [Dependencies](#dependencies)
    + [Required](#required)
    + [Optional](#optional)
- [Setup](#setup)
- [Usage](#usage)
    + [Fingerprinting in Parallel](#fingerprint-parallel) 

<a name="dependencies"></a>
##Dependencies##

<a name="required"></a>
###Required###
The following is a non-comprehensive list of necessary Python dependencies:
- [Numpy](http://www.numpy.org/)
- [RDKit](http://www.rdkit.org/)
- [mmh3](https://pypi.python.org/pypi/mmh3)
- [h5py](https://pypi.python.org/pypi/h5py/2.5.0)
- [seaware_academic](https://github.com/keiserlab/seaware-academic
- [python_utilities](https://github.com/sdaxen/python_utilities)

<a name="optional"></a>
###Optional###
- [mpi4py](http://mpi4py.scipy.org/)
- [futures](https://pypi.python.org/pypi/futures)

<a name="setup"></a>
##Setup##

Clone this repository into a directory that is in your PYTHONPATH.

<a name="usage"></a>
##Usage##

In a python script, type: 
```python
import e3fp
```

<a name="fingerprint-parallel"></a>
###Fingerprinting in Parallel###

```python
from python_utilities.parallel import Parallelizer
from python_utilities.scripting import setup_logging
from e3fp.sea_utils.util import smiles_to_dict, lists_dicts_to_molecules, fprint_params_to_fptype
from e3fp.pipeline import native_tuples_from_smiles

# setup
setup_logging()
parallelizer = Parallelizer(parallel_mode="processes")
smiles_file = "smiles_file.smi"

# read in space-delimited smiles, name from file
smiles_dict = smiles_to_dict(smiles_file)
smiles_iter = ((smiles, name) for name, smiles in smiles_dict.iteritems())

# setup conformer generation, fingerprinting, and folding parameters
confgen_kwargs = {"pool_multiplier": 2, "rmsd_cutoff": 0.5}
fprint_kwargs = {"shell_radius": 1.671, "stereo": True, "include_disconnected": True}
fold_kwargs = {"bits": 1024}
kwargs = {"save": False, "first": 8, "level": 5,
          "confgen_kwargs": confgen_kwargs, "fprint_kwargs": fprint_kwargs,
          "fold_kwargs": fold_kwargs}

# fingerprint in parallel
mol_list_dict = {data[1]:result for result, data in parallelizer.run_gen(native_tuples_from_smiles, smiles_iter, kwargs=kwargs)}

# save to SEA molecules file
fp_type = fprint_params_to_fptype(fold_level=fold_kwargs["bits"], level=kwargs["level"], **fprint_kwargs)
lists_dicts_to_molecules("molecules.csv.gz", smiles_dict, mol_list_dict, fp_type)
```

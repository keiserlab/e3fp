#E3FP: Extended 3-Dimensional FingerPrint#

E3FP is a 3D molecular fingerprinting method inspired by Extended Connectivity FingerPrints (ECFP)<sup>[1](#rogers2010)</sup>.

##Table of Contents##
- [Dependencies](#dependencies)
    + [Required](#required)
    + [Optional](#optional)
- [Setup](#setup)
- [Usage](#usage)
    + [Examples](#examples)
- [References](#references)

<a name="dependencies"></a>
##Dependencies##

<a name="required"></a>
###Required###
A non-comprehensive list of necessary Python dependencies:
- [Numpy](http://www.numpy.org/)
- [RDKit](http://www.rdkit.org/)
- [mmh3](https://pypi.python.org/pypi/mmh3)
- [h5py](https://pypi.python.org/pypi/h5py/2.5.0)
- [seaware-academic](https://github.com/keiserlab/seaware-academic)
- [python_utilities](https://github.com/sdaxen/python_utilities)

<a name="optional"></a>
###Optional###
Optional dependencies for parallelization:
- [mpi4py](http://mpi4py.scipy.org/)
- [futures](https://pypi.python.org/pypi/futures)

<a name="setup"></a>
##Setup##

Clone this repository into a directory that is in your PYTHONPATH.

<a name="usage"></a>
##Usage##

To use E3FP in a python script, enter: 
```python
import e3fp
```

<a name="examples"></a>
###Examples###

See [examples]("./examples") directory for example scripts that show how to perform several common tasks with E3FP:
- `examples/parallel_fingerprinting.py`: This example shows how to use [python_utilities](https://github.com/sdaxen/python_utilities) as well as the E3FP pipeline, given SMILES, to generate molecules, build a conformer library, fingerprint the conformers, fold fingerprints, and convert to ascii strings for SEA searching in a single step, using all available CPUs.
- `examples/library_generation.py`: This example shows how to convert SEA targets files prepared for ECFP to E3FP targets files; it then demonstrates how to create a new E3FP library with fit generation.
- `examples/sea_searching.py`: This examples demonstrates searching already fingerprinted molecules against a SEA library.

<a name="references"></a>
##References##
<a name="rogers2010"></a>
1. Rogers, D. & Hahn, M. Extended-connectivity fingerprints. *J. Chem. Inf. Model.* **50**, 742-54 (2010).

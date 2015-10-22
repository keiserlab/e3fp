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
- [Numpy](http://www.numpy.org)\*
- [RDKit](http://www.rdkit.org)
- [CairoCFFI](https://github.com/SimonSapin/cairocffi)\*
- [mmh3](https://pypi.python.org/pypi/mmh3)\*
- [h5py](https://pypi.python.org/pypi/h5py/2.5.0)\*
- [seaware-academic](https://github.com/keiserlab/seaware-academic)
- [python_utilities](https://github.com/sdaxen/python_utilities)

<a name="optional"></a>
###Optional###
Optional dependencies:
- parallelization:
    + [mpi4py](http://mpi4py.scipy.org)\*
    + [futures](https://pypi.python.org/pypi/futures)\*
- molecular standardisation:
    + [standardiser](https://wwwdev.ebi.ac.uk/chembl/extra/francis/standardiser)
- protonation states:
    + [cxcalc](https://docs.chemaxon.com/display/CALCPLUGS/cxcalc+command+line+tool)

\*_Install with_ `pip install <package_name>`

<a name="setup"></a>
##Setup##

1. Download this repository to your machine
    - Clone this repository to your machine with `git clone https://github.com/keiserlab/e3fp.git`
    - OR download an archive by navigating to [https://github.com/keiserlab/e3fp](https://github.com/keiserlab/e3fp) and clicking "Download ZIP". Extract the archive.
2. Add the path to the repository to your `$PYTHONPATH`. On Unix, this can be done with `export PYTHONPATH=[PATH/TO/REPO]:$PYTHONPATH` where `[PATH/TO/REPO]` is replaced with the path on your machine.

<a name="usage"></a>
##Usage##

To use E3FP in a python script, enter: 
```python
import e3fp
```

<a name="examples"></a>
###Examples###

See [examples](examples) directory for example scripts that show how to perform several common tasks with E3FP:
- [`parallel_fingerprinting.py`](examples/parallel_fingerprinting.py): This example shows how to use [python_utilities](https://github.com/sdaxen/python_utilities) as well as the E3FP pipeline, given SMILES, to generate molecules, build a conformer library, fingerprint the conformers, fold fingerprints, and convert to ascii strings for SEA searching in a single step, using all available CPUs. When fingerprinting many molecules, it is recommended to split conformer generation and fingerprinting into two separate steps, as a small percentage of molecules may hang at conformer generation, using up the CPUs.
- [`library_generation.py`](examples/library_generation.py): This example shows how to convert SEA targets files prepared for ECFP to E3FP targets files; it then demonstrates how to create a new E3FP SEA library with background fit generation.
- [`sea_searching.py`](examples/sea_searching.py): This example demonstrates searching already fingerprinted molecules against a SEA<sup>[2](#keiser2007)</sup> library.

<a name="references"></a>
##References##
<a name="rogers2010"></a>
1. Rogers, D. & Hahn, M. Extended-connectivity fingerprints. *J. Chem. Inf. Model.* **50**, 742-54 (2010).
<a name="keiser2007"></a>
2. Keiser, M.J. et al. Relating protein pharmacology by ligand chemistry. *Nat. Biotech.* **25** (2), 197-206 (2007).


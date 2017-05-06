# E3FP: Extended 3-Dimensional FingerPrint

E3FP is a 3D molecular fingerprinting method inspired by Extended Connectivity FingerPrints (ECFP)<sup>[1](#rogers2010)</sup>.

## Table of Contents
- [Dependencies](#dependencies)
    + [Required](#required)
    + [Optional](#optional)
- [Setup and Installation](#setup)
- [Usage](#usage)
    + [Examples](#examples)
- [References](#references)

<a name="dependencies"></a>
## Dependencies

`e3fp` is compatible with Python 2.x. It additionally has the following
dependencies:

<a name="required"></a>
### Required
- [NumPy](https://www.numpy.org)
- [SciPy](https://www.scipy.org)
- [RDKit](http://www.rdkit.org)
- [mmh3](https://pypi.python.org/pypi/mmh3)
- [python_utilities](https://github.com/sdaxen/python_utilities)

<a name="optional"></a>
### Optional
The following packages are required for the specified features:
- parallelization:
    + [mpi4py](http://mpi4py.scipy.org)
    + [futures](https://pypi.python.org/pypi/futures)
- molecular standardisation:
    + [standardiser](https://wwwdev.ebi.ac.uk/chembl/extra/francis/standardiser)
- protonation states:
    + [cxcalc](https://docs.chemaxon.com/display/CALCPLUGS/cxcalc+command+line+tool)

<a name="setup"></a>
## Setup and Installation

The following installation approaches are listed in order of recommendation.

### Clone the repository
0. Install above dependencies.
1. Download this repository to your machine.
    - Clone this repository to your machine with `git clone https://github.com/keiserlab/e3fp.git`.
    - OR download an archive by navigating to [https://github.com/keiserlab/e3fp](https://github.com/keiserlab/e3fp) and clicking "Download ZIP". Extract the archive.
2. Add the path to the repository to your `$PYTHONPATH`. On Unix, this can be done with `export PYTHONPATH=[PATH/TO/REPO]:$PYTHONPATH` where `[PATH/TO/REPO]` is replaced with the path on your machine.

<a name="usage"></a>
## Usage

To use E3FP in a python script, enter: 
```python
import e3fp
```

<a name="references"></a>
## References
<a name="rogers2010"></a>
1. Rogers, D. & Hahn, M. Extended-connectivity fingerprints. *J. Chem. Inf. Model.* **50**, 742-54 (2010). doi: [10.1021/ci100050t](http://dx.doi.org/10.1021/ci100050t)

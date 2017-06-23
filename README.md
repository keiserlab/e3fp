[![Build Status](https://travis-ci.org/keiserlab/e3fp.svg?branch=master)](https://travis-ci.org/keiserlab/e3fp)

# E3FP: Extended 3-Dimensional FingerPrint

E3FP<sup>[1](#axen2017)</sup> is a 3D molecular fingerprinting method inspired by Extended Connectivity FingerPrints (ECFP)<sup>[2](#rogers2010)</sup>.

## Table of Contents
- [Dependencies](#dependencies)
    + [Required](#required)
    + [Optional](#optional)
- [Setup and Installation](#setup)
- [Usage and Overview](#usage)
- [References](#references)

<a name="dependencies"></a>
## Dependencies

`e3fp` is compatible with Python 2.7.x and 3.6.x. It additionally has the following
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
- storing conformer energies:
    + [h5py](http://www.h5py.org/)

<a name="setup"></a>
## Setup and Installation

The following installation approaches are listed in order of recommendation.
Each of these approaches first requires an installation of [RDKit](http://www.rdkit.org).

### Option 1: Install with Conda
E3FP is on the [Anaconda distribution](https://docs.continuum.io/anaconda).
Conda is a cross-platform package manager. This approach is highly recommended
as it installs *all* required packages.
```bash
conda create -c keiserlab --name e3fp_env e3fp
```
To get the latest version of E3FP, follow Option 3 below.

### Option 2: Install with Pip
1. Install with
```bash
pip install e3fp
```
2. To install the optional Python dependencies, run
```bash
pip install mpi4py futures standardiser h5py
```

### Option 3: Clone the repository
0. Install any of the optional dependencies above.
1. Download this repository to your machine.
    - Clone this repository to your machine with `git clone https://github.com/keiserlab/e3fp.git`.
    - OR download an archive by navigating to [https://github.com/keiserlab/e3fp](https://github.com/keiserlab/e3fp) and clicking "Download ZIP". Extract the archive.
2. Install with
    ```bash
    cd e3fp
    python setup.py build_ext --inplace
    python setup.py install
    ```

### Testing
After installation, it is recommended to run all tests with `nose`,

```bash
pip install nose
nosetests e3fp
```

<a name="usage"></a>
## Usage and Overview

To use E3FP in a python script, enter: 
```python
import e3fp
```
See [`pipeline.py`](e3fp/pipeline.py) for methods for generating conformers and E3FP fingerprints
from various inputs.

Run `python e3fp/conformer/generate.py --help` for options for generating conformers.

Run `python e3fp/fingerprint/generate.py --help` for options for generating E3FP fingerprints.

See [`defaults.cfg`](e3fp/config/defaults.cfg) for an example params file.

See the [E3FP paper repo](https://github.com/keiserlab/e3fp-paper) for an application of E3FP
and all code used for the E3FP paper<sup>[1](#axen2017)</sup>.

<a name="references"></a>
## References
<a name="axen2017"></a>
1. Axen SD, Huang XP, Caceres EL, Gendelev L, Roth BL, Keiser MJ. A Simple Representation Of Three-Dimensional Molecular Structure. *bioRxiv* (2017). doi: [10.1101/136705](http://dx.doi.org/10.1101/136705). *(preprint)*
<a name="rogers2010"></a>
2. Rogers D & Hahn M. Extended-connectivity fingerprints. *J. Chem. Inf. Model.* **50**, 742-54 (2010). doi: [10.1021/ci100050t](http://dx.doi.org/10.1021/ci100050t)

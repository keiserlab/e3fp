|Build Status|

E3FP: Extended 3-Dimensional FingerPrint
========================================

E3FP [1]_ is a 3D molecular fingerprinting method
inspired by Extended Connectivity FingerPrints
(ECFP) [2]_.

Table of Contents
-----------------

-  Dependencies_

   -  Required_
   -  Optional_

-  `Setup and Installation`_
-  `Usage and Overview`_
-  References_

Dependencies
------------

``e3fp`` is compatible with Python 2.7.x and 3.x. It additionally has
the following dependencies:

Required
~~~~~~~~

- NumPy_
- SciPy_
- RDKit_
- mmh3_
- python_utilities_

Optional
~~~~~~~~

The following packages are required for the specified
features:

- parallelization:

  + mpi4py_
  + futures_

- molecular standardisation:

  + standardiser_

- protonation states:

  + cxcalc_

- storing conformer energies:

  + h5py_

Setup and Installation
----------------------

The following installation approaches are listed in order of
recommendation. Each of these approaches first requires an installation
of RDKit_.

Option 1: Install with Conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

E3FP is on the `Anaconda distribution`_. Conda is a
cross-platform package manager. This approach is highly recommended as
it installs *all* required packages.

.. code:: bash

    conda create -c keiserlab -c rdkit -c sdaxen --name e3fp_env e3fp

To get the latest version of E3FP, follow Option 3 below.

Option 2: Install with Pip
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install with

   .. code:: bash

       pip install e3fp

2. To install the optional Python dependencies, run

   .. code:: bash

       pip install mpi4py futures standardiser h5py

Option 3: Clone the repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

0. Install any of the optional dependencies above.
1. Download this repository to your machine.

   -  Clone this repository to your machine with

      .. code:: bash

          git clone https://github.com/keiserlab/e3fp.git

   -  OR download an archive by navigating to
      https://github.com/keiserlab/e3fp and clicking "Download ZIP".
      Extract the archive.

2. Install with

   .. code:: bash

       cd e3fp
       python setup.py build_ext --inplace
       python setup.py install

Testing
~~~~~~~

After installation, it is recommended to run all tests with ``nose``,

.. code:: bash

    pip install nose
    nosetests e3fp

Usage and Overview
------------------

To use E3FP in a python script, enter:

.. code:: python

    import e3fp

See |pipeline.py|_ for methods for generating
conformers and E3FP fingerprints from various inputs.

Run ``python e3fp/conformer/generate.py --help`` for options for
generating conformers.

Run ``python e3fp/fingerprint/generate.py --help`` for options for
generating E3FP fingerprints.

See |defaults.cfg|_ for an example
params file.

See the `E3FP paper repo`_ for an application of E3FP and all code used
for the E3FP paper [1]_.

References
----------
.. [1] Axen SD, Huang XP, Caceres EL, Gendelev L, Roth BL, Keiser MJ.
  A Simple Representation Of Three-Dimensional Molecular Structure.
  *J. Med. Chem.* (2017).
  doi: `10.1021/acs.jmedchem.7b00696 <http://dx.doi.org/10.1021/acs.jmedchem.7b00696>`_.
  |F1000 recommended|
.. [2] Rogers D & Hahn M.
  Extended-connectivity fingerprints.
  *J. Chem. Inf. Model.* **50**, 742-54 (2010).
  doi: `10.1021/ci100050t <http://dx.doi.org/10.1021/ci100050t>`_

.. URL's
.. _E3FP paper repo: https://github.com/keiserlab/e3fp-paper
.. _RDKit: http://www.rdkit.org
.. _NumPy: https://www.numpy.org
.. _SciPy: https://www.scipy.org
.. _mmh3: https://pypi.python.org/pypi/mmh3
.. _python_utilities: https://github.com/sdaxen/python_utilities
.. _mpi4py: http://mpi4py.scipy.org
.. _futures: https://pypi.python.org/pypi/futures
.. _standardiser: https://wwwdev.ebi.ac.uk/chembl/extra/francis/standardiser
.. _cxcalc: https://docs.chemaxon.com/display/CALCPLUGS/cxcalc+command+line+tool
.. _h5py: http://www.h5py.org/
.. _Anaconda distribution: https://docs.continuum.io/anaconda

.. Images/Substitutions
.. |pipeline.py| replace:: ``pipeline.py``
.. _pipeline.py: e3fp/pipeline.py
.. |defaults.cfg| replace:: ``defaults.cfg``
.. _defaults.cfg: e3fp/config/defaults.cfg
.. |Build Status| image:: https://travis-ci.org/keiserlab/e3fp.svg?branch=master
   :target: https://travis-ci.org/keiserlab/e3fp
.. |F1000 recommended| image:: http://cdn.f1000.com.s3.amazonaws.com/images/badges/badgef1000.gif
   :target: http://f1000.com/prime/727824514?bd=1
   :alt: Access the recommendation on F1000Prime
   :width: 120px
   :scale: 75 %

Setup and Installation
======================

.. contents::


Dependencies
------------

E3FP is compatible with Python 2.7.x and 3.x. It additionally has
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


Installation
------------

The following installation approaches are listed in order of
recommendation. Each of these approaches first requires an installation
of RDKit_.

Option 1: Install with Conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

E3FP is on the `Anaconda distribution`_. Conda is a
cross-platform package manager. This approach is highly recommended as
it installs *all* required packages.

.. code:: bash

    $ conda create -c keiserlab -c rdkit -c sdaxen --name e3fp_env e3fp

To get the latest version of E3FP, follow Option 3 below.

Option 2: Install with Pip
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install with

   .. code:: bash

       $ pip install e3fp

2. To install the optional Python dependencies, run

   .. code:: bash

       $ pip install mpi4py futures standardiser h5py

Option 3: Clone the repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

0. Install any of the optional dependencies above.
1. Download this repository to your machine.

   -  Clone this repository to your machine with

      .. code:: bash

          $ git clone https://github.com/keiserlab/e3fp.git

   -  OR download an archive by navigating to
      https://github.com/keiserlab/e3fp and clicking "Download ZIP".
      Extract the archive.

2. Install with

   .. code:: bash

       $ cd e3fp
       $ python setup.py build_ext --inplace
       $ python setup.py install


Testing
-------

After installation, it is recommended to run all tests with ``nose``,

.. code:: bash

    $ pip install nose
    $ nosetests e3fp


.. URLs
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

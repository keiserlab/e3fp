Setup and Installation
======================

Dependencies
------------

E3FP is compatible with Python 2.7.x and 3.x. It additionally has the
following dependencies:

Required
~~~~~~~~

- NumPy_
- SciPy_
- RDKit_
- mmh3_
- python_utilities_

Optional
~~~~~~~~

The following packages are required for the specified features:

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

The following installation approaches are listed in order of recommendation.
All but the first of these approaches requires a prior installation of RDKit_.

Option 1: Install with Conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

E3FP is on the `Anaconda distribution`_. Conda is a cross-platform package
manager. This approach is highly recommended as it installs *all* required
packages.

.. code:: bash

    $ conda create -c keiserlab -c rdkit -c sdaxen --name e3fp_env e3fp

.. warning:: Due to a
  `potential bug <https://www.mail-archive.com/rdkit-discuss@lists.sourceforge.net/msg07315.html>`_
  in recent versions of conda, rdkit may not import on some systems. If this is
  the case, simply downgrade conda to v4.3.25 before creating the above
  environment:
  
  .. code:: bash

    conda install conda=4.3.25

To get the latest version of E3FP, follow :ref:`Option 3: Clone the Repository`.

Option 2: Install with Pip
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install with

   .. code:: bash

       $ pip install e3fp

2. To install the optional Python dependencies, run

   .. code:: bash

       $ pip install mpi4py futures standardiser h5py

Option 3: Clone the Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

0. Install any of the optional dependencies above.
1. Download this repository to your machine.

   -  Clone this repository to your machine with

      .. code:: bash

          $ git clone https://github.com/keiserlab/e3fp.git

   -  OR download an archive by navigating to the repository_ and clicking
      "Download ZIP". Extract the archive.

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


.. include:: substitutions.rst

Setup and Installation
======================

Dependencies
------------

E3FP is compatible with Python 3.x. It additionally has the
following dependencies:

Required
~~~~~~~~

- NumPy_
- SciPy_
- RDKit_
- mmh3_
- python_utilities_
- smart_open_

Optional
~~~~~~~~

The following packages are required for the specified features:

- parallelization:

  + mpi4py_

- molecular standardisation:

  + standardiser_

- protonation states:

  + cxcalc_

- storing conformer energies:

  + h5py_

- faster fingerprint metric calculations:

  + numba_


Installation
------------

The following installation approaches are listed in order of recommendation.

Option 1: Install with Pip
~~~~~~~~~~~~~~~~~~~~~~~~~~

Basic installation:

.. code:: bash

    $ pip install e3fp

With optional dependencies:

.. code:: bash

    $ pip install e3fp[optional]


Option 2: Install from conda-forge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

E3FP is available on conda-forge.

.. code:: bash

    $ conda create -n e3fp_env -c conda-forge e3fp
    $ conda activate e3fp_env

To install optional dependencies:

.. code:: bash

    $ conda install -c conda-forge mpi4py h5py standardiser

Option 3: Install from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Clone the repository:

   .. code:: bash

      $ git clone https://github.com/keiserlab/e3fp.git
      $ cd e3fp

2. Install for development in an already-activated environment.

   You can do this using pip:

   .. code:: bash

      $ pip install -e .[dev]

  Or use uv_ to set up a development environment:

  .. code:: bash

      $ uv sync --extra dev

Testing
-------

Run tests using pytest:

.. code:: bash

    $ pip install pytest  # if not already installed
    $ pytest e3fp


.. include:: substitutions.rst

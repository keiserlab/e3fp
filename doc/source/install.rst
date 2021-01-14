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

1. Install with

   .. code:: bash

       $ conda create -c conda-forge --name e3fp_env e3fp
       $ conda activate e3fp_env

2. To install the optional Python dependencies, run

   .. code:: bash

       $ conda install -c conda-forge futures mpi4py h5py standardiser

To get the latest version of E3FP, follow :ref:`Option 3: Clone the Repository`.

Option 2: Install with Pip
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install with

   .. code:: bash

       $ pip install e3fp

2. To install the optional Python dependencies, run

   .. code:: bash

       $ pip install futures mpi4py h5py standardiser

Option 3: Clone the Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Download this repository to your machine.

   -  Clone this repository to your machine with

      .. code:: bash

          $ git clone https://github.com/keiserlab/e3fp.git
          $ cd e3fp

   -  OR download an archive by navigating to the repository_ and clicking
      "Download ZIP". Extract the archive.

2. Install the optional dependencies and any required ones using pip or conda.

  .. note:: The easiest way to install the dependencies is with

    .. code:: bash

        $ conda env create --name e3fp_env --file environment.yml
        $ conda activate e3fp_env

3. Install with

   .. code:: bash

       $ python setup.py build_ext --inplace
       $ python setup.py install

Testing
-------

After installation, it is recommended to run all tests with ``pytest``.
After running :code:`pip install pytest` or :code:`conda install -c conda-forge pytest`, run

.. code:: bash

    $ pytest e3fp


.. include:: substitutions.rst

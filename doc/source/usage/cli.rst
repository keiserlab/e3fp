.. _cli:

Command Line Interface
======================

Command line interfaces (CLI) are provided for the two most common tasks:
conformer generation and fingerprinting. At the moment, using the CLI requires
:ref:`downloading the E3FP source <repo_download>`.

In the below examples, we assume the E3FP repository is located at
``$E3FP_REPO``.


.. _cli_confgen:

Conformer Generation
--------------------

To see all available options, run

.. code:: bash

    $ python $E3FP_REPO/e3fp/conformer/generate.py --help

We will generate conformers for the molecule whose SMILES string is defined in
``caffeine.smi``.

.. literalinclude:: ../examples/data/caffeine.smi
   :caption: caffeine.smi

The below example generates at most 3 conformers for this molecule.

.. code:: bash

    $ python $E3FP_REPO/e3fp/conformer/generate.py -s caffeine.smi --num_conf 3 -o ./

The result is a multi-conformer SDF file called ``caffeine.sdf.bz2`` in the
current directory.


.. _cli_fprint:

Fingerprinting
--------------------

To see all available options, run

.. code:: bash

    $ python $E3FP_REPO/e3fp/fingerprint/generate.py --help

To continue the above example, we will fingerprint our caffeine conformers.

.. code:: bash

    $ python $E3FP_REPO/e3fp/fingerprint/generate.py caffeine.sdf.bz2 --bits 1024

The result is a file ``fingerprints.fps.bz2`` containing a
:py:class:`.FingerprintDatabase`. To use such a database, consult
:ref:`fp_storage`.

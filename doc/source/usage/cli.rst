Command Line Interface
======================

Command line interfaces (CLI) are provided for the two most common tasks:
conformer generation and fingerprinting. At the moment, using the CLI requires
:ref:`downloading the E3FP source <Option 3: Clone the Repository>`.

In the below examples, we assume the E3FP repository is located at
``$E3FP_REPO``.

Conformer Generation CLI
------------------------

To see all available options, run

.. command-output:: python $E3FP_REPO/e3fp/conformer/generate.py --help
   :shell:

We will generate conformers for the molecule whose SMILES string is defined in
``caffeine.smi``.

.. literalinclude:: ../examples/data/caffeine.smi
   :caption: caffeine.smi

The below example generates at most 3 conformers for this molecule.

.. code-block:: shell-session

    $ python $E3FP_REPO/e3fp/conformer/generate.py -s caffeine.smi --num_conf 3 -o ./
    2017-07-17 00:11:05,743|WARNING|Only 1 processes available. 'mpi' mode not available.
    2017-07-17 00:11:05,748|INFO|num_proc is not specified. 'processes' mode will use all 8 processes
    2017-07-17 00:11:05,748|INFO|Parallelizer initialized with mode 'processes' and 8 processors.
    2017-07-17 00:11:05,748|INFO|Input type: Detected SMILES file(s)
    2017-07-17 00:11:05,748|INFO|Input file number: 1
    2017-07-17 00:11:05,748|INFO|Parallel Type: processes
    2017-07-17 00:11:05,748|INFO|Out Directory: ./
    2017-07-17 00:11:05,749|INFO|Overwrite Existing Files: False
    2017-07-17 00:11:05,749|INFO|Target Conformer Number: 3
    2017-07-17 00:11:05,749|INFO|First Conformers Number: all
    2017-07-17 00:11:05,749|INFO|Pool Multiplier: 1
    2017-07-17 00:11:05,749|INFO|RMSD Cutoff: 0.5
    2017-07-17 00:11:05,749|INFO|Maximum Energy Difference: None
    2017-07-17 00:11:05,749|INFO|Forcefield: UFF
    2017-07-17 00:11:05,749|INFO|Starting.
    2017-07-17 00:11:05,779|INFO|Generating conformers for caffeine.
    2017-07-17 00:11:05,823|INFO|Generated 1 conformers for caffeine.
    2017-07-17 00:11:05,829|INFO|Saved conformers for caffeine to ./caffeine.sdf.bz2.

The result is a multi-conformer SDF file called ``caffeine.sdf.bz2`` in the
current directory.

Fingerprinting CLI
------------------

To see all available options, run

.. command-output:: python $E3FP_REPO/e3fp/fingerprint/generate.py --help
   :shell:

To continue the above example, we will fingerprint our caffeine conformers.

.. code-block:: shell-session

    $ python $E3FP_REPO/e3fp/fingerprint/generate.py caffeine.sdf.bz2 --bits 1024
    2017-07-17 00:12:33,797|WARNING|Only 1 processes available. 'mpi' mode not available.
    2017-07-17 00:12:33,801|INFO|num_proc is not specified. 'processes' mode will use all 8 processes
    2017-07-17 00:12:33,801|INFO|Parallelizer initialized with mode 'processes' and 8 processors.
    2017-07-17 00:12:33,801|INFO|Initializing E3FP generation.
    2017-07-17 00:12:33,801|INFO|Getting SDF files
    2017-07-17 00:12:33,801|INFO|SDF File Number: 1
    2017-07-17 00:12:33,802|INFO|Database File: fingerprints.fpz
    2017-07-17 00:12:33,802|INFO|Max First Conformers: 3
    2017-07-17 00:12:33,802|INFO|Bits: 1024
    2017-07-17 00:12:33,802|INFO|Level/Max Iterations: 5
    2017-07-17 00:12:33,802|INFO|Shell Radius Multiplier: 1.718
    2017-07-17 00:12:33,802|INFO|Stereo Mode: True
    2017-07-17 00:12:33,802|INFO|Connected-only mode: on
    2017-07-17 00:12:33,802|INFO|Invariant type: Daylight
    2017-07-17 00:12:33,802|INFO|Parallel Mode: processes
    2017-07-17 00:12:33,802|INFO|Starting
    2017-07-17 00:12:33,829|INFO|Generating fingerprints for caffeine.
    2017-07-17 00:12:33,935|INFO|Generated 1 fingerprints for caffeine.
    2017-07-17 00:12:34,011|INFO|Saved FingerprintDatabase with fingerprints to fingerprints.fpz

The result is a file ``fingerprints.fpz`` containing a
:py:class:`.FingerprintDatabase`. To use such a database, consult
:ref:`Fingerprint Storage`.

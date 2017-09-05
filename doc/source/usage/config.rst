.. _config:

Configuration
=============

.. contents::

E3FP configurational parameters are stored in the widely used INI_ file format.
These may be passed to :ref:`cli` programs or parsed to Python dicts for 
:ref:`pipeline_methods` or other lower-level functions.

Loading Default Parameters
--------------------------

The below example shows all default parameters, accessed via the
:py:mod:`e3fp.config` module.

.. literalinclude:: ../../../e3fp/config/defaults.cfg
   :caption: `defaults.cfg <https://github.com/keiserlab/e3fp/blob/documentation/e3fp/config/defaults.cfg>`_

:py:mod:`configparser` is used internally to parse and store these
config parameters.

.. code:: python

   >>> from e3fp.config.params import default_params
   >>> default_params
   <ConfigParser.SafeConfigParser instance at 0x10639eea8>
   >>> print(default_params.sections())
   ['preprocessing', 'conformer_generation', 'fingerprinting']
   >>> default_params.items('fingerprinting')
   [('bits', '1024'),
    ('level', '5'),
    ('first', '3'),
    ('radius_multiplier', '1.718'),
    ('stereo', 'True'),
    ('counts', 'False'),
    ('include_disconnected', 'True'),
    ('rdkit_invariants', 'False'),
    ('merge_duplicate_substructs', 'True'),
    ('exclude_floating', 'True')]

Parsing User-Provided Parameters
--------------------------------

A user may provide a custom config file.

.. literalinclude:: ../examples/data/new_params.cfg
   :caption: new_params.cfg

When parsing, we may specify that all missing parameters are set to defaults.

.. code:: python

   >>> from e3fp.config.params import read_params
   >>> config = read_params("source/examples/data/new_params.cfg", fill_defaults=True)
   >>> config.items('fingerprinting')
   [('bits', '4096'),
    ('level', '5'),
    ('first', '10'),
    ('radius_multiplier', '1.718'),
    ('stereo', 'True'),
    ('counts', 'False'),
    ('include_disconnected', 'True'),
    ('rdkit_invariants', 'False'),
    ('merge_duplicate_substructs', 'True'),
    ('exclude_floating', 'True')]

Converting Parameters to Argument Dicts
---------------------------------------

To pass the parameters to Python methods for fingerprinting and conformer
generation, we need to convert them to Python dicts.

.. code:: python

   >>> from e3fp.pipeline import params_to_dicts
   >>> confgen_params, fprint_params = params_to_dicts(config)
   >>> fprint_params
   {'bits': 4096,
    'counts': False,
    'exclude_floating': True,
    'first': 10,
    'include_disconnected': True,
    'level': 5,
    'merge_duplicate_substructs': True,
    'radius_multiplier': 1.718,
    'rdkit_invariants': False,
    'stereo': True}

.. _INI: https://en.wikipedia.org/wiki/INI_file


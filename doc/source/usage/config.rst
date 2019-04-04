Configuration
=============

E3FP configurational parameters are stored in the widely used INI_ file
format. These may be passed to :ref:`Command Line Interface` programs or
parsed to Python dicts for :ref:`Pipeline Methods` or other lower-level
functions.

Loading Default Parameters
--------------------------

The below example shows all default parameters, accessed via the
:py:mod:`e3fp.config` module.

.. literalinclude:: ../../../e3fp/config/defaults.cfg
   :caption: `defaults.cfg <https://github.com/keiserlab/e3fp/blob/master/e3fp/config/defaults.cfg>`_

:py:mod:`configparser` is used internally to parse and store these
config parameters.

   >>> from e3fp.config.params import default_params
   >>> default_params
   <ConfigParser.SafeConfigParser instance at 0x...>
   >>> print(default_params.sections())
   ['preprocessing', 'conformer_generation', 'fingerprinting']
   >>> default_params.items('fingerprinting')
   [('bits', '1024'), ('level', '5'), ('first', '3'), ('radius_multiplier', '1.718'), ('stereo', 'True'), ('counts', 'False'), ('include_disconnected', 'True'), ('rdkit_invariants', 'False'), ('merge_duplicate_substructs', 'True'), ('exclude_floating', 'True')]

Parsing User-Provided Parameters
--------------------------------

A user may provide a custom config file.

.. literalinclude:: ../examples/data/new_params.cfg
   :caption: new_params.cfg

.. doctest::

   >>> from e3fp.config.params import read_params
   >>> config = read_params("source/examples/data/new_params.cfg")
   >>> config.items('fingerprinting')
   [('bits', '4096'), ('first', '10')]

When passing these parameters to any downstream methods, default options will
be used except where these options are specified.

Converting Parameters to Argument Dicts
---------------------------------------

To pass the parameters to Python methods for fingerprinting and conformer
generation, we need to convert them to Python dicts.

   >>> from e3fp.pipeline import params_to_dicts
   >>> confgen_params, fprint_params = params_to_dicts(config)
   >>> fprint_params
   {'bits': 4096, 'first': 10}

.. _INI: https://en.wikipedia.org/wiki/INI_file


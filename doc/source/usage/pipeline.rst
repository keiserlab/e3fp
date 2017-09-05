.. _pipeline:

Pipeline
========

In some cases, one will want to patch E3FP into an existing pipeline. We
therefore provide useful methods in the :py:mod:`.pipeline` module. Each of
these methods wraps functionality in other modules for converting various
intermediary inputs to outputs.

As fingerprinting many molecules is embarrassingly parallel, we highly
recommend employing a parallelization strategy. We use our own
python_utilities_ module.

.. include:: ../substitutions.rst

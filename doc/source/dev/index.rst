Developer Notes
===============

We welcome contributions to E3FP! These notes are designed to help developers
contribute code

Authoring Code
--------------

Code Formatting
~~~~~~~~~~~~~~~

E3FP's code should be *readable*. To ensure this, we rigorously follow the
PEP8_ style conventions and PEP257_ docstring conventions, which maximize
readability of the code and ease of future development. You may check your
code for conformation to these conventions with the pycodestyle_ and
pydocstyle_ utilities, respectively. Where the code is necessarily
complicated, inline comments should reorient the reader.

Utility Methods and Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three sets of utility methods and classes are provided: `e3fp.util`,
`e3fp.conformer.util`, and `e3fp.fingerprint.util`. These provide general and
often-used functionality in their corresponding packages. Additionally, they
provide E3FP-specific errors and exceptions.

Warnings and Errors
~~~~~~~~~~~~~~~~~~~

By default, warnings in Python are silent. We therefore provide a warning base
class `e3fp.util.E3FPWarning` that is not silent by default. We provide several
general warnings:

:py:class:`.E3FPDeprecationWarning`
    warns when a deprecated method is called or class is instantiated.

    .. seealso::

       :ref:`Deprecation`

:py:class:`.E3FPEfficiencyWarning`
    warns when a method, module version, or combination of parameters is known
    to be inefficient.

    .. note::

       If possible, the warning message should advise on a more efficient
       approach.

E3FP-specific errors should inherit `e3fp.util.E3FPError` base class. Several
fingerprinting-specific errors are defined in `e3fp.fingerprint.util`.

Deprecation
~~~~~~~~~~~

Whenever changing the interface or behavior of a user-facing method or class,
it is proper to deprecate it for at least one release, so that the users have
time to update their scripts accordingly. A deprecated method should providing
an `e3fp.util.E3FPDeprecationWarning`, notifying the user in which release to
expect the method or class to be removed, and updating the documentation
accordingly. This functionality is automated with the `e3fp.util.deprecated`
decorator, as shown in this example:

    >>> import sys
    >>> sys.stderr = sys.stdout
    >>> from e3fp.util import deprecated
    >>> @deprecated("1.1", remove_version="1.3", msg="Function no longer needed.")
    ... def deprecated_method():
    ...     """A method to demonstrate method deprecation."""
    ...     pass
    >>> deprecated_method()
    ...: E3FPDeprecationWarning: Function `my_function` was deprecated in 1.1 and will be removed in 1.3. Function no longer needed.

In the api documentation, the method will appear as:

.. function:: deprecated_method()

    .. note:: Deprecated in e3fp 1.1.
       `deprecated_method` will be removed in e3fp 1.3. Function no longer needed.

    A method to demonstrate method deprecation.

.. note::
    If no `remove_version` is specified, then the remove version defaults to the
    next release after deprecation. For example, if the method was deprecated in
    1.1, it is by default marked for removal in 1.2.

Contributing Code
~~~~~~~~~~~~~~~~~

Before contributing code to E3FP, it is advisable for major modifications to 
submit an issue to the
`issue tracker`_ to enable other
developers to contribute to the design of the code and to reduce the amount of
work necessary to conform the code to E3FP's standards. After writing the code,
create a `pull request`_. This is best even if you have push access to the
E3FP repo, as it enables the test suite to be run on the new code prior to
merging it with the remaining code base.

Writing Tests
~~~~~~~~~~~~~

The standard in E3FP is to commit a test for new functionality simultaneously
with the new functionality or within the same pull request. While this slows
development, it prevents building a large backlog of untested methods and
classes. 

These should ideally be unit tests, though for some complicated
functionalities, such as fingerprinting, integration tests are also
necessary. For these complicated functions, specific units may still be
tested using :py:mod:`unittest.mock`. For example,
:py:meth:`unittest.mock.patch` may be used to force a high level method to
produce a specific output. For examples, see the `fingeprinting tests
<https://github.com/keiserlab/e3fp/blob/master/e3fp/test/test_fingerprinting.py>`_.

Continuous Integration
~~~~~~~~~~~~~~~~~~~~~~

E3FP uses `GitHub Actions`_ for continuous integration. This ensures that each commit
and pull request passes all tests on a variety of a systems and for all
supported versions of Python. Additionally, GitHub Actions updates code coverage on
Codecov_ and tests all usage examples in the documentation using `doctest`.

Documentation
-------------

In general, it is best to document the rationale and basic usage of a module,
class, or method in its docstring instead of in a separate documentation file.
See, for example, the docstring for `e3fp.fingerprint.db.FingerprintDatabase`.
We use a variety of tools to ensure that our documentation is always
up-to-date. The official documentation is hosted on ReadtheDocs_ and is
automatically generated when new code is committed to the repository.

Documenting Code
~~~~~~~~~~~~~~~~

E3FP uses NumPy's `docstring conventions`_ for all docstrings. These are
parsed by Sphinx_ using Napoleon_. All usage examples must be fully
functional, as these are tested using `doctest`.

The purpose of a docstring is to explain the purpose of a class/method, any
relevant implementation details, its parameters, its attributes, its outputs,
and its usage. The goal is clarity. For self-evident methods with descriptive
variables, a simple one- ine summary is all that is needed. For complicated use
cases, often involving other methods/classes, it is better to document the
usage elsewhere in the documentation.

Documentation Usage
~~~~~~~~~~~~~~~~~~~

Coming soon.

.. todo::
    Write documentation usage

Releasing Code
--------------

.. todo::
    Write release protocol

.. _PEP8: https://www.python.org/dev/peps/pep-0008/
.. _PEP257: https://www.python.org/dev/peps/pep-0257/
.. _pycodestyle: http://pycodestyle.pycqa.org/en/latest/
.. _pydocstyle: http://pydocstyle.pycqa.org/en/latest/
.. _docstring conventions: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
.. _Napoleon: http://www.sphinx-doc.org/en/stable/ext/napoleon.html
.. _Sphinx: http://www.sphinx-doc.org/en/stable/index.html
.. _doctest: https://docs.python.org/3/library/doctest.html
.. _pull request: https://help.github.com/articles/creating-a-pull-request/
.. _GitHub Actions: https://github.com/keiserlab/e3fp/actions
.. _Codecov: https://codecov.io/github/keiserlab/e3fp

.. include:: ../substitutions.rst

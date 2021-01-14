E3FP: Extended 3-Dimensional FingerPrint
========================================

|Docs Status| |CI Status| |Codecov Status| |PyPi Version| |Conda Version| |License|

E3FP [1]_ is a 3D molecular fingerprinting method inspired by Extended
Connectivity FingerPrints (ECFP) [2]_, integrating tightly with the RDKit_.

Documentation is hosted by ReadTheDocs_, and development occurs on GitHub_.

Installation and Usage
----------------------

For installation and usage instructions, see the
`documentation <http://e3fp.readthedocs.io>`__.

See the E3FP `paper repository`_ for an application of E3FP and all code used
for the E3FP paper [1]_.

License
-------

E3FP is available under the `GNU Lesser General Public License version 3.0
<https://www.gnu.org/licenses/lgpl.html>`_ (LGPLv3). See the
`documentation <http://e3fp.readthedocs.io/en/latest/overview.html#license>`__
for more details.


References
----------

.. [1] |axen2017|
.. [2] |rogers2010|

.. substitutions

.. _RDKit: http://www.rdkit.org
.. _GitHub: https://github.com/keiserlab/e3fp
.. _paper repository: https://github.com/keiserlab/e3fp-paper
.. _ReadTheDocs: http://e3fp.readthedocs.io
.. |axen2017_doi| image:: https://img.shields.io/badge/doi-10.1021/acs.jmedchem.7b00696-blue.svg
    :target: http://dx.doi.org/10.1021/acs.jmedchem.7b00696
    :alt: Access the paper
.. |axen2017| replace:: Axen SD, Huang XP, Caceres EL, Gendelev L, Roth BL, Keiser MJ. A Simple Representation Of Three-Dimensional Molecular Structure. *J. Med. Chem.* **60** (17): 7393–7409 (2017). |axen2017_doi| |bioRxiv| |F1000 recommended|
.. |rogers2010_doi| image:: https://img.shields.io/badge/doi-10.1021/ci100050t-blue.svg
    :target: http://dx.doi.org/10.1021/ci100050t
    :alt: Access the paper
.. |rogers2010| replace:: Rogers D & Hahn M. Extended-connectivity fingerprints. *J. Chem. Inf. Model.* **50**: 742-54 (2010). |rogers2010_doi|
.. |CI Status| image:: https://github.com/keiserlab/e3fp/workflows/CI/badge.svg?branch=master
   :target: https://github.com/keiserlab/e3fp/actions?query=workflow%3ACI
   :alt: CI Status
.. |Docs Status| image:: http://readthedocs.org/projects/e3fp/badge/?version=latest
   :target: http://e3fp.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. |Codecov Status| image:: https://codecov.io/github/keiserlab/e3fp/coverage.svg?branch=master
   :target: https://codecov.io/github/keiserlab/e3fp?branch=master
   :alt: Code Coverage
.. |PyPi Version| image:: https://img.shields.io/pypi/v/e3fp.svg
   :target: https://pypi.python.org/pypi/e3fp
   :alt: Package on PyPi
.. |Conda Version| image:: https://img.shields.io/conda/v/conda-forge/e3fp.svg
   :target: https://anaconda.org/conda-forge/e3fp
   :alt: Package on Anaconda
.. |License| image:: https://img.shields.io/badge/license-LGPLv3-blue.svg
   :target: https://github.com/keiserlab/e3fp/blob/master/LICENSE.txt
.. |F1000 recommended| image:: http://cdn.f1000.com.s3.amazonaws.com/images/badges/badgef1000.gif
   :target: http://f1000.com/prime/727824514?bd=1
   :alt: Access the recommendation on F1000Prime
   :width: 120px
   :scale: 75 %
.. |bioRxiv| image:: https://img.shields.io/badge/bioRxiv-136705-blue.svg
    :target: https://doi.org/10.1101/136705
    :alt: Access the preprint on bioRxiv

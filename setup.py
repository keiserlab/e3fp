import os

from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np
from e3fp import version

ON_RTD = os.environ.get("READTHEDOCS") == "True"


requirements = [
    "scipy>=0.18.0",
    "numpy>=1.11.3",
    "mmh3>=2.3.1",
    "cython>=0.25.2",
    "sdaxen_python_utilities>=0.1.4",
]
if ON_RTD:  # ReadTheDocs can't handle C libraries
    requirements = requirements[-1:] + ["mock"]

test_requirements = ["nose", "mock"]

classifiers = [
    "Programming Language :: Python",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Cython",
    "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
    "Operating System :: OS Independent",
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Topic :: Scientific/Engineering :: Chemistry",
    "Topic :: Software Development :: Libraries :: Python Modules",
]


def get_readme():
    with open("README.rst") as f:
        return f.read()

ext_modules = [
    Extension(
        "e3fp.fingerprint.metrics._fast",
        sources=["e3fp/fingerprint/metrics/_fast.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="e3fp",
    packages=[
        "e3fp",
        "e3fp.config",
        "e3fp.conformer",
        "e3fp.fingerprint",
        "e3fp.test",
    ],
    version=version,
    description="Molecular 3D fingerprinting",
    long_description=get_readme(),
    keywords="e3fp 3d molecule fingerprint conformer",
    author="Seth Axen",
    author_email="seth.axen@gmail.com",
    license="LGPLv3",
    url="https://github.com/keiserlab/e3fp",
    classifiers=classifiers,
    download_url="https://github.com/keiserlab/e3fp/tarball/" + version,
    install_requires=requirements,
    include_package_data=True,
    test_suite="nose.collector",
    tests_require=test_requirements,
    ext_modules=cythonize(ext_modules),
)

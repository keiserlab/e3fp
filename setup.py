try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

requirements = [
    'scipy>=0.18.0',
    'numpy>=1.11.3',
    'mmh3>=2.3.1',
    'sdaxen_python_utilities>=0.1.4',
]

test_requirements = ['nose', 'mock']

classifiers = ['Programming Language :: Python',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.6',
               'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
               'Operating System :: OS Independent',
               'Development Status :: 4 - Beta',
               'Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'Topic :: Scientific/Engineering :: Chemistry',
               'Topic :: Software Development :: Libraries :: Python Modules'
               ]

setup(
    name='e3fp',
    packages=['e3fp', 'e3fp.config', 'e3fp.conformer', 'e3fp.fingerprint',
              'e3fp.test'],
    version='1.0',
    description='Molecular 3D fingerprinting',
    keywords='e3fp 3d molecule fingerprint conformer',
    author='Seth Axen',
    author_email='seth.axen@gmail.com',
    license='LGPLv3',
    url='https://github.com/keiserlab/e3fp',
    classifiers=classifiers,
    download_url='https://github.com/keiserlab/e3fp/tarball/1.0',
    install_requires=requirements,
    include_package_data=True,
    test_suite='nose.collector',
    tests_require=test_requirements,
)

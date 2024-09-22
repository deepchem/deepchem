import sys
import time
from setuptools import setup, find_packages

if '--release' in sys.argv:
    IS_RELEASE = True
    sys.argv.remove('--release')
else:
    # Build a nightly package by default.
    IS_RELEASE = False

# Environment-specific dependencies.
extras = {
    'jax': ['jax', 'jaxlib', 'dm-haiku', 'optax'],
    'torch': ['torch==2.2.1', 'torchvision', 'pytorch-lightning', 'dgl<2.2.1', 'dgllife'],
    'tensorflow': ['tensorflow', 'tensorflow_probability', 'tensorflow_addons'],
    'dqc': ['dqc', 'xitorch', 'torch==2.2.1', 'pylibxc2']
}


# get the version from deepchem/__init__.py
def _get_version():
    with open('deepchem/__init__.py') as fp:
        for line in fp:
            if line.startswith('__version__'):
                g = {}
                exec(line, g)
                base = g['__version__']
                if IS_RELEASE:
                    return base
                else:
                    # nightly version : .devYearMonthDayHourMinute
                    if base.endswith('.dev') is False:
                        # Force to add `.dev` if `--release` option isn't passed when building
                        base += '.dev'
                    return base + time.strftime("%Y%m%d%H%M%S")

        raise ValueError('`__version__` not defined in `deepchem/__init__.py`')


setup(name='deepchem',
      version=_get_version(),
      url='https://github.com/deepchem/deepchem',
      maintainer='DeepChem contributors',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Information Technology',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
      ],
      license='MIT',
      description='Deep learning models for drug discovery, \
        quantum chemistry, and the life sciences.',
      keywords=[
          'deepchem',
          'chemistry',
          'biology',
          'materials-science',
          'life-science',
          'drug-discovery',
      ],
      packages=find_packages(exclude=["*.tests"]),
      project_urls={
          'Documentation': 'https://deepchem.readthedocs.io/en/latest/',
          'Source': 'https://github.com/deepchem/deepchem',
      },
      install_requires=[
          'joblib',
          'numpy<2',
          'pandas',
          'scikit-learn',
          'sympy',
          'scipy>=1.10.1',
          'rdkit',
      ],
      extras_require=extras,
      python_requires='>=3.7,<3.12')

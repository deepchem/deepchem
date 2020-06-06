from setuptools import setup

config = {'setup_requires': ['pbr'], 'pbr': True}
setup(**config)
from setuptools import setup, find_packages

setup(name='deepchem',
      packages=find_packages())

from setuptools import setup, find_packages
import sys

if 'develop' not in sys.argv:
    raise NotImplementedError("Use python setup.py develop.")
setup(name="deep_chem",
   url='https://github.com/pandegroup/deep-learning',
   description='Deep Learning Toolchain for Cheminformatics and Protein Analysis',
   install_requires=['keras'],
   packages=find_packages())

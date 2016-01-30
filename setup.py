from setuptools import setup, find_packages
import sys

setup(name="deepchem",
   url='https://github.com/pandegroup/deep-learning',
   description='Deep Learning Toolchain for Cheminformatics and Protein Analysis',
   install_requires=['keras'],
   packages=find_packages())

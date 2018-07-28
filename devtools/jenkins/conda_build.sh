#!/bin/bash
cd devtools/conda-recipe
source activate base
conda upgrade conda -y
conda install conda-build anaconda-client conda-verify -y
conda upgrade conda-build anaconda-client conda-verify -y
conda build purge
export python_version=2.7
conda build deepchem -c defaults -c rdkit -c omnia -c conda-forge
export python_version=3.5
conda build deepchem -c defaults -c rdkit -c omnia -c conda-forge
export python_version=3.6
conda build deepchem -c defaults -c rdkit -c omnia -c conda-forge

export package_name=deepchem-gpu
export tensorflow_enabled=tensorflow-gpu
export python_version=2.7
conda build deepchem -c defaults -c rdkit -c omnia -c conda-forge
export python_version=3.5
conda build deepchem -c defaults -c rdkit -c omnia -c conda-forge
export python_version=3.6
conda build deepchem -c defaults -c rdkit -c omnia -c conda-forge

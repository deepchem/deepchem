#!/usr/bin/env bash
# Used to make a conda environment with deepchem

# This line is needed for using conda activate
# This command is nearly equal to `conda init` command
source $(conda info --root)/etc/profile.d/conda.sh

if [ -z "$python_version" ]
then
    echo "Using python 3.6 by default"
    export python_version=3.6
else
    echo "Using python "$python_version". But recommended to use python 3.6."
fi

if [ -z "$1" ];
then
    echo "Installing DeepChem in current env"
else
    export envname=$1
    conda create -y --name $envname python=$python_version
    conda activate $envname
fi

yes | pip install --upgrade pip
conda install -y -q -c deepchem -c rdkit -c conda-forge -c omnia \
    biopython \
    cloudpickle=1.4.1 \
    mdtraj \
    networkx \
    openmm \
    pdbfixer \
    pillow \
    py-xgboost \
    rdkit \
    simdna \
    pymatgen \
    pytest \
    pytest-cov \
    flaky
yes | pip install pyGPGO
yes | pip install -U matminer tensorflow==2.2 tensorflow-probability==0.10

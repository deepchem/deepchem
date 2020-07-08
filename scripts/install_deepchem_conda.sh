#!/usr/bin/env bash
# Used to make a conda environment with deepchem

# This line is needed for using conda activate
# This command is nearly equal to `conda init` command
source $(conda info --root)/etc/profile.d/conda.sh

## Force python version 3.6 because other versions will
## throw errors
export python_version=3.6

if [ -z "$1" ];
then
    echo "Installing DeepChem in current env"
    read -r -p "Are you sure? [y/N] " response
    response=${response,,}
    if [[ "$response" =~ ^(yes|y)$ ]];
    then
        echo "Continuing..."
    else
        echo "Quitting without changes"
        exit 1
    fi
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
    flaky \
    ipython
yes | pip install pyGPGO
yes | pip install -U matminer tensorflow==2.2 tensorflow-probability==0.10
conda update scikit-learn 

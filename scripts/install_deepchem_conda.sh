#!/usr/bin/env bash
# Used to make a conda environment with deepchem

# Change commented out line For gpu tensorflow
#export tensorflow=tensorflow-gpu
export tensorflow=tensorflow


if [ -z "$1" ]
then
    echo "Must Specify Conda Environment Name"
fi

if [ -z "$python_version" ]
then
    echo "Using python 3.5 by default"
    export python_version=3.5
fi

export envname=$1
conda create -y --name $envname python=$python_version
source activate $envname
conda install -y -c omnia pdbfixer=1.4
conda install -y -c rdkit rdkit
conda install -y -c conda-forge joblib=0.11
conda install -y -c conda-forge six
conda install -y -c conda-forge mdtraj
conda install -y -c conda-forge scikit-learn=0.18.1
conda install -y -c conda-forge setuptools
conda install -y -c conda-forge keras=1.2.2
conda install -y -c conda-forge networkx=1.11
conda install -y -c conda-forge xgboost=0.6a2
conda install -y -c conda-forge pillow=4.2.1
conda install -y -c conda-forge pandas=0.19.2
conda install -y -c conda-forge $tensorflow=1.2.1
conda install -y -c conda-forge nose=1.3.7
conda install -y -c conda-forge nose-timer=0.7.0
conda install -y -c conda-forge flaky=3.3.0

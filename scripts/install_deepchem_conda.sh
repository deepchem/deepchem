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
conda install -y joblib
conda install -c anaconda six
conda install -y -c omnia mdtraj
conda install -y scikit-learn=0.18.1
conda install -y setuptools
conda install -y -c conda-forge keras=1.2.2
conda install -y -c anaconda networkx=1.11
conda install -y -c conda-forge xgboost=0.6a2
conda install -y -c anaconda pillow=4.2.1
conda install -y -c anaconda pandas=0.19.2
conda install -y $tensorflow=1.2.0
conda install -y -c anaconda nose
conda install -c omnia nose-timer
conda install -c spyder-ide flaky=3.3.0

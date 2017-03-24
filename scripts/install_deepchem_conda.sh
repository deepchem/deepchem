#!/usr/bin/env bash
# Used to make a conda environment with deepchem

# Change commented out line For gpu tensorflow
export tensorflow=tensorflow-gpu
#export tensorflow=tensorflow

if [ -z "$1" ]
then
    echo "Must Specify Conda Environment Name"
fi

export envname=$1
conda create -y --name $envname python=3.5
source activate $envname
conda install -y -c omnia openbabel=2.4.0
conda install -y -c omnia pdbfixer=1.4
conda install -y -c rdkit rdkit
conda install -y joblib
yes | pip install six
conda install -y -c omnia mdtraj
conda install -y scikit-learn
conda install -y setuptools
conda install -y -c conda-forge keras=1.2.2
conda install -y -c conda-forge protobuf=3.1.0
conda install -y -c anaconda networkx=1.11
yes | pip install $tensorflow==1.0.1
yes | pip install nose

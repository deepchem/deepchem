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

if [ "$python_version" == "3.5" ]
then
    export protobuf_url=https://storage.googleapis.com/tensorflow/linux/cpu/protobuf-3.1.0-cp35-none-linux_x86_64.whl
else
    export protobuf_url=https://storage.googleapis.com/tensorflow/linux/cpu/protobuf-3.1.0-cp27-none-linux_x86_64.whl
fi

export envname=$1
conda create -y --name $envname python=$python_version
source activate $envname
conda install -y -c omnia pdbfixer=1.4
conda install -y -c rdkit rdkit
conda install -y joblib
yes | pip install six
conda install -y -c omnia mdtraj
conda install -y scikit-learn
conda install -y setuptools
conda install -y -c conda-forge keras=1.2.2
yes | pip install --upgrade $protobuf_url
yes | pip install --upgrade $protobuf_url
conda install -y -c anaconda networkx=1.11
conda install -y -c conda-forge xgboost=0.6a2
conda install -y -c pillow
conda install -y -c anaconda pandas=0.19.2
conda install -y $tensorflow=1.1.0
yes | pip install nose
yes | pip install nose-timer
yes | pip install flaky==3.3.0

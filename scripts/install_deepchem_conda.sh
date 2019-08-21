#!/usr/bin/env bash
# Used to make a conda environment with deepchem

# Change commented out line For gpu tensorflow
#export tensorflow=tensorflow-gpu
export tensorflow=tensorflow

if [ -z "$gpu" ]
then
    export tensorflow=tensorflow
    echo "Using Tensorflow (CPU MODE) by default."
elif [ "$gpu" == 1 ]
then
    export tensorflow=tensorflow-gpu
    echo "Using Tensorflow (GPU MODE)."
else
    echo "Using Tensorflow (CPU MODE) by default."
fi

if [ -z "$python_version" ]
then
    echo "Using python 3.5 by default"
    export python_version=3.5
else
    echo "Using python "$python_version". But recommended to use python 3.5."
fi

if [ -z "$1" ];
then
    echo "Installing DeepChem in current env"
else
    export envname=$1
    conda create -y --name $envname python=$python_version
    source activate $envname
fi

unamestr=`uname`
if [[ "$unamestr" == 'Darwin' ]]; then
   source activate root
   conda install -y -q conda=4.3.25
   source activate $envname
fi

yes | pip install --upgrade pip
conda install -y -q -c deepchem -c rdkit -c conda-forge -c omnia \
    mdtraj \
    pdbfixer \
    rdkit \
    joblib \
    six \
    scikit-learn \
    networkx \
    pillow \
    pandas \
    nose \
    nose-timer \
    flaky \
    zlib \
    requests \
    xgboost \
    simdna \
    pbr \
    setuptools \
    biopython \
    numpy
yes | pip install $tensorflow==1.14.1
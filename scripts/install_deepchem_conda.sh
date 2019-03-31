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

conda install -y -q -c omnia pdbfixer=1.4
yes | pip install --upgrade pip
conda install -y -q -c deepchem mdtraj=1.9.1
conda install -y -q -c rdkit rdkit=2017.09.1
conda install -y -q -c conda-forge joblib=0.11 \
    six=1.11.0 \
    scikit-learn=0.19.1 \
    networkx=2.1 \
    pillow=5.0.0 \
    pandas=0.22.0 \
    nose=1.3.7 \
    nose-timer=0.7.0 \
    flaky=3.3.0 \
    zlib=1.2.11 \
    requests=2.18.4 \
    xgboost=0.6a2 \
    simdna=0.4.2 \
    pbr=3.1.1 \
    setuptools=39.0.1 \
    biopython=1.71 \
    numpy=1.14
yes | pip install $tensorflow==1.13.1

#!/usr/bin/env bash
# Used to make a conda environment with deepchem

# Change commented out line For gpu tensorflow
#export tensorflow=tensorflow-gpu
export tensorflow=tensorflow


if [ -z "$python_version" ]
then
    echo "Using python 3.6 by default"
    export python_version=3.6
fi

if [ -z "$1" ];
then
    echo "Installing DeepChem in current env"
else
    export envname=$1
    conda create -y --name $envname python=$python_version
    source activate $envname
fi

conda install -y -q -c omnia pdbfixer=1.4
conda install -y -q -c deepchem mdtraj=1.9.1
conda install -y -q -c rdkit rdkit=2018.03.3.0
conda install -y -q -c conda-forge joblib=0.12 \
    six=1.11.0 \
    scikit-learn=0.19.1 \
    networkx=2.1 \
    pillow=5.1.0 \
    pandas=0.23.3 \
    nose=1.3.7 \
    nose-timer=0.7.3 \
    flaky=3.4.0 \
    requests=2.18.4 \
    xgboost=0.72.1 \
    simdna=0.4.2 \
    jupyter=1.0.0 \
    pbr=4.2.0 \
    biopython=1.72
yes | pip install $tensorflow==1.9.0

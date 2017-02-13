#/bin/bash
envname=`cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 16 | head -n 1`
conda create --name $envname
source activate $envname
conda install -c omnia openbabel=2.4.0
conda install -c rdkit rdkit
conda install joblib
pip install six
conda install -c omnia mdtraj
conda install scikit-learn
conda install setuptools
conda install keras
conda install -c conda-forge protobuf=3.1.0
pip install tensorflow-gpu
pip install nose
python setup.py install

cd examples
python benchmark.py -d tox21

source deactivate
conda remove --name $envname --all
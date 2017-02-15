#!/usr/bin/env bash
envname=`cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 16 | head -n 1`
conda create --name $envname python=3.5
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
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-linux_x86_64.whl
pip install --upgrade $TF_BINARY_URL
pip install nose
python setup.py install

cd examples
python benchmark.py -d tox21
cd ..
nosetests -v devtools/jenkins/compare_results.py --with-xunit || true

source deactivate
conda remove --name $envname --all
rm results.csv
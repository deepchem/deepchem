#!/usr/bin/env bash
envname=`cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 16 | head -n 1`
sed -i -- 's/tensorflow$/tensorflow-gpu/g' scripts/install_deepchem_conda.sh
export python_version=2.7
bash scripts/install_deepchem_conda.sh $envname
source activate $envname
python setup.py install
conda install -y jupyter
conda install -y nbconvert
conda install -y jupyter_client
conda install -y ipykernel
conda install -y matplotlib
yes | pip install nglview
conda install -y ipywidgets
conda install -y zlib
conda install -y cmake
yes | pip install gym[atari]
conda install -y -c auto pubchempy
conda install -y xlrd
conda install -y seaborn

cd examples/notebooks
nosetests --with-timer tests.py --with-xunit --xunit-file=notebook_tests.xml|| true

source deactivate
# conda remove --name $envname --all

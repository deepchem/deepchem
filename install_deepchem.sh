!#/bin/bash
wget https://repo.continuum.io/archive/Anaconda3-4.3.0-Linux-x86_64.sh -O anaconda.sh
bash anaconda.sh -b -p $HOME/anaconda
export PATH="$HOME/anaconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda config --add channels http://conda.binstar.org/omnia
bash scripts/install_deepchem_conda.sh deepchem
source activate deepchem
pip install yapf==0.16.0
pip install coveralls
python setup.py develop

sudo apt-get update
wget http://repo.continuum.io/archive/Anaconda2-2.4.1-Linux-x86_64.sh -O anaconda.sh;
bash anaconda.sh -b -p $HOME/anaconda
hash -r
conda config --set always_yes yes --set changeps1 no
conda config --add channels http://conda.binstar.org/omnia
conda update -q conda
conda info -a
conda install pandas
conda install -c omnia rdkit
conda install -c omnia openbabel
conda install joblib
conda install -c omnia theano
conda install -c omnia keras
python setup.py install

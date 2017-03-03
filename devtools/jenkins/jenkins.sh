#!/usr/bin/env bash
envname=`cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 16 | head -n 1`
conda create --name $envname python=3.5
bash scripts/install_deepchem_conda.sh $envname
source activate $envname
python setup.py install

rm examples/results.csv || true

cd examples/pdbbind
bash get_pdbbind.sh
cd ..

cd examples
python benchmark.py -d pdbbind
cd ..
nosetests -v devtools/jenkins/compare_results.py --with-xunit || true

source deactivate
conda remove --name $envname --all
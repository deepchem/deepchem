#!/usr/bin/env bash
envname=`cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 16 | head -n 1`
sed -i -- 's/tensorflow$/tensorflow-gpu/g' scripts/install_deepchem_conda.sh
export python_version=2.7
bash scripts/install_deepchem_conda.sh $envname
source activate $envname
python setup.py install

# Run adme test
cd examples/
nosetests --with-timer tests.py --with-xunit --xunit-file=example_tests.xml|| true

source deactivate
conda remove --name $envname --all

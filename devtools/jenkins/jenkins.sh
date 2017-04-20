#!/usr/bin/env bash
envname=`cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 16 | head -n 1`
conda create --name $envname python=3.5
bash scripts/install_deepchem_conda.sh $envname
source activate $envname
python setup.py install

rm examples/results.csv || true
cd contrib/atomicconv/core
python opt_random_tensorgraph.py


source deactivate
conda remove --name $envname --all
export retval=$(($retval1 + $retval2))
exit ${retval}

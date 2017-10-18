#!/usr/bin/env bash
envname=`cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 16 | head -n 1`
sed -i -- 's/tensorflow$/tensorflow-gpu/g' scripts/install_deepchem_conda.sh
bash scripts/install_deepchem_conda.sh $envname
source activate $envname
python setup.py install

rm examples/results.csv || true
cd examples
python benchmark.py -d tox21 -m weave -m graphconv -m tf_robust -m tf -m irv -m xgb -m logreg --seed 123
export retval_tox21=$?
python benchmark.py -d hiv -m weave -m graphconv -m tf -m irv -m logreg --seed 123
export retval_hiv=$?
python benchmark.py -d delaney -m weave_regression -m graphconreg -m krr -m tf_regression -m xgb_regression -m mpnn --seed 123
export retval_delaney=$?
python benchmark.py -d qm7 -m dtnn -m graphconvreg -m krr_ft -m krr -m tf_regression_ft -m tf_regression --seed 123
export retval_qm7=$?

cd ..
nosetests -v devtools/jenkins/compare_results.py --with-xunit || true
export retval=$?

source deactivate
conda remove --name $envname --all

#!/usr/bin/env bash
envname=`cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 16 | head -n 1`
sed -i -- 's/tensorflow$/tensorflow-gpu/g' scripts/install_deepchem_conda.sh
bash scripts/install_deepchem_conda.sh $envname
source activate $envname
python setup.py install

rm examples/results.csv || true
export DEEPCHEM_DATA_DIR='/tmp/molnet_'$envname
mkdir $DEEPCHEM_DATA_DIR
cd examples
python benchmark.py -d tox21 -m weave -m graphconv -m tf_robust -m tf -m irv -m xgb -m logreg --seed 123
python benchmark.py -d clintox -m weave -m graphconv -m tf -m tf_robust -m irv -m xgb -m logreg --seed 123
python benchmark.py -d sider -m weave -m graphconv -m tf -m tf_robust -m irv -m logreg --seed 123
python benchmark.py -d bbbp -m weave -m graphconv -m tf -m irv -m logreg --seed 123

python benchmark.py -d hiv -m weave -m graphconv -m tf -m irv -m logreg --seed 123
python benchmark.py -d muv -m weave -m graphconv -m tf -m tf_robust -m irv -m logreg --seed 123
python benchmark.py -d bace -m weave -m graphconv -m tf -m irv -m logreg --seed 123

python benchmark.py -d delaney -m weave_regression -m graphconvreg -m tf_regression -m dag_regression -m mpnn --seed 123
python benchmark.py -d sampl -m weave_regression -m graphconvreg -m tf_regression -m dag_regression -m mpnn --seed 123
python benchmark.py -d lipo -m weave_regression -m graphconvreg -m tf_regression -m dag_regression -m mpnn --seed 123

python benchmark.py -d qm7 -m dtnn -m graphconvreg -m tf_regression_ft -m tf_regression -s random -s stratified --seed 123
python benchmark.py -d qm7b -m dtnn -m tf_regression_ft --seed 123
python benchmark.py -d qm8 -m dtnn -m mpnn -m graphconvreg -m tf_regression -m tf_regression_ft --seed 123
python benchmark.py -d qm9 -m dtnn -m graphconvreg -m tf_regression -m tf_regression_ft --seed 123

cd ..
nosetests -v devtools/jenkins/compare_results.py --with-xunit || true
export retval2=$?

nosetests -a 'slow' --with-timer deepchem --with-xunit --xunit-file=slow_tests.xml|| true
export retval3=$?

source deactivate
conda remove --name $envname --all

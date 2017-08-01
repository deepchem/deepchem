#!/usr/bin/env bash
envname=`cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 16 | head -n 1`
sed -i -- 's/tensorflow$/tensorflow-gpu/g' scripts/install_deepchem_conda.sh
export python_version=2.7
bash scripts/install_deepchem_conda.sh $envname
source activate $envname
python setup.py install
conda install jupyter
conda install nbconvert
conda install jupyter_client
conda install ipykernel
conda install matplotlib
pip install nglview
conda install ipywidgets

cd examples/notesbooks
jupyter-nbconvert --to notebook --execute --ExecutePreprocessor.timeout=60 --output tmp.ipynb protein_ligand_complex_notebook.ipynb

nosetests --with-timer tests.py --with-xunit -unit-file=notebook_tests.xml|| true

source deactivate
conda remove --name $envname --all
export retval=$(($retval1 + $retval2 + $retval3))
return ${retval}
#!/usr/bin/env bash
# Create the docs and push them to S3
# -----------------------------------
envname=`cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 16 | head -n 1`
sed -i -- 's/tensorflow$/tensorflow-gpu/g' scripts/install_deepchem_conda.sh
bash scripts/install_deepchem_conda.sh $envname
source activate $envname
python setup.py install

echo "About to install numpydoc, s3cmd"
pip install -I sphinx==1.3.5 sphinx_bootstrap_theme
pip install numpydoc s3cmd msmb_theme sphinx_rtd_theme nbsphinx delegator.py
conda install -y -q jupyter
conda install -y -q matplotlib

cd examples/notebooks
python ../../devtools/jenkins/convert_to_rst.py
cd ../..

mkdir -p docs/_build
echo "About to build docs"
sphinx-apidoc -f -o docs/source deepchem
sphinx-build -b html docs/source docs/_build
# Copy 
cp -r docs/_build/ website/docs/
echo "About to push docs to s3"
python devtools/jenkins/push-docs-to-s3.py
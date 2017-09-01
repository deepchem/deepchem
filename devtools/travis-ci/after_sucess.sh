#!/usr/bin/env bash
echo '$TRAVIS_PULL_REQUEST $TRAVIS_BRANCH'
echo $TRAVIS_PULL_REQUEST $TRAVIS_BRANCH

if [[ "$TRAVIS_PULL_REQUEST" != "false" ]]; then
    echo "This is a pull request. No deployment will be done."; exit 0
fi

if [[ "$TRAVIS_BRANCH" != "master" ]]; then
    echo "No deployment on BRANCH='$TRAVIS_BRANCH'"; exit 0
fi

# Create the docs and push them to S3
# -----------------------------------
echo "About to install numpydoc, s3cmd"
pip install -I sphinx==1.3.5 sphinx_bootstrap_theme
pip install numpydoc s3cmd msmb_theme sphinx_rtd_theme nbsphinx
conda list -e
mkdir -p docs/_build
echo "About to build docs"
sphinx-apidoc -f -o docs/source deepchem
sphinx-build -b html docs/source docs/_build
# Copy 
cp -r docs/_build/ website/docs/
echo "About to push docs to s3"
python devtools/travis-ci/push-docs-to-s3.py

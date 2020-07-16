#!/bin/bash -e

if [ $TRAVIS_PYTHON_VERSION == '3.7' ]; then
    find ./deepchem -name "*.py" ! -name '*load_dataset_template.py' | xargs python -m doctest -v;
fi

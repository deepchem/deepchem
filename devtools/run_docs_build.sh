#!/bin/bash -e

if [ $TRAVIS_PYTHON_VERSION == '3.7' ]; then
  cd docs && pip install -r requirements.txt;
  make clean html && cd ..;
fi

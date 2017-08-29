#!/usr/bin/env bash
if [ "$package_name" == "deepchem" ] || [ "$package_name" == "deepchem-gpu" ]
then
    $PYTHON setup.py install
fi

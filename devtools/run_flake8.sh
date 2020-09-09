#!/bin/bash -e

items=(
    "deepchem/hyper"
    "deepchem/dock"
    "deepchem/metrics"
    "deepchem/data"
    "deepchem/splits"
    "deepchem/feat"
    "deepchem/utils"
)

for item in "${items[@]}" ; do
  flake8 ${item} --count --show-source --statistics
done

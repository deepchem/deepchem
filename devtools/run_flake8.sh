#!/bin/bash -e

items=(
    "deepchem/data"
    "deepchem/dock"
    "deepchem/feat"
    "deepchem/hyper"
    "deepchem/metalearning"
    "deepchem/metrics"
    "deepchem/rl"
    "deepchem/splits"
    "deepchem/trans"
    "deepchem/utils"
)

for item in "${items[@]}" ; do
  flake8 ${item} --count --show-source --statistics
done

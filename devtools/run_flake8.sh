#!/bin/bash -e

items=(
    "deepchem/hyper"
    "deepchem/dock"
    "deepchem/metrics"
    "deepchem/data"
)

for item in "${items[@]}" ; do
  flake8 ${item} --count --show-source --statistics
done

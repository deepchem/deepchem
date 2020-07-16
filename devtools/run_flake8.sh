#!/bin/bash -e

items=(
    "deepchem/hyper"
)

for item in "${items[@]}" ; do
  flake8 ${item} --count --show-source --statistics
done

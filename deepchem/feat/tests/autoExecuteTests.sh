#!/usr/bin/env bash
# Runs all the tests in featurize
for file in ./*; do
  if [ ${file: -3} == ".py" ]; then
    pytest "$file"
  fi
done

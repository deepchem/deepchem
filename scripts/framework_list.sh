#!/bin/bash

# Initialize lists for each framework
pytorch_files=""
jax_files=""
tensorflow_files=""
no_framework_files=""

# Recursively check all Python files in the 'deepchem' package
while IFS= read -r -d '' file; do
  if grep -q -E "import torch|from torch" "$file"; then
    pytorch_files+="$file "
  elif grep -q -E "import jax|from jax" "$file"; then
    jax_files+="$file "
  elif grep -q -E "import tensorflow|from tensorflow" "$file"; then
    tensorflow_files+="$file "
  else
    no_framework_files+="$file "
  fi
done < <(find deepchem -type f -name "*.py" -print0)

# Export lists as environment variables
export PYTORCH_FILES="$pytorch_files"
export JAX_FILES="$jax_files"
export TENSORFLOW_FILES="$tensorflow_files"
export NO_FRAMEWORK_FILES="$no_framework_files"
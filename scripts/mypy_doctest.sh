#!/bin/bash

module_to_search="$1"
mypy_or_pytest="$2"

while IFS= read -r -d '' file; do
  if grep -q -E "import $module_to_search|from $module_to_search" "$file"; then
    if [ "$mypy_or_pytest" = "mypy" ]; then
      mypy --ignore-missing-imports "$file"
    elif [ "$mypy_or_pytest" = "doctest" ]; then
      DGLBACKEND=pytorch pytest --doctest-modules --doctest-continue-on-failure "$file" 
    else
      echo "Invalid argument. Please use 'mypy' or 'pytest'."
      exit 1
    fi
  fi
done < <(find deepchem -type f -name "*.py" -print0)
# This script may work on only Bash and Zsh
# usage: source scripts/flake8_for_ci.sh

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
    "deepchem/molnet"
)

for item in "${items[@]}" ; do
    flake8 ${item} --count --show-source --statistics
done

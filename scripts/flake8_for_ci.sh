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
    "deepchem/models"
)

for item in "${items[@]}" ; do
    echo ${item}; flake8 ${item} --exclude=__init__.py --count --show-source --statistics
done

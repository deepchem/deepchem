#!/usr/bin/env bash
# Used to make a conda environment with deepchem

# This line is needed for using conda activate
# This command is nearly equal to `conda init` command
source $(conda info --root)/etc/profile.d/conda.sh

if [ -z "$python_version" ]
then
    echo "Using python 3.6 by default"
    export python_version=3.6
else
    echo "Using python "$python_version". But recommended to use python 3.6."
fi

if [ -z "$1" ];
then
    echo "Installing DeepChem in current env"
else
    export envname=$1
    conda create -y --name $envname python=$python_version
    conda activate $envname
fi

# Install dependencies except PyTorch Geometric
conda env update --file $PWD/requirements.yml
pip install -r $PWD/requirements.txt
pip install -r $PWD/requirements-test.txt

# For PyTorch
list=(`cat $PWD/requirements-torch.txt | xargs`)
for pkg in "${list[@]}" ; do
    pkg=`echo ${pkg} | sed -e "s/[\r\n]\+//g"`
    pip install ${pkg}+cpu -f https://download.pytorch.org/whl/torch_stable.html
done

# For PyTorch Geometric
export TORCH=1.5.0
list=(`cat $PWD/requirements-pyg.txt | xargs`)
for pkg in "${list[@]}" ; do
    pkg=`echo ${pkg} | sed -e "s/[\r\n]\+//g"`
    if [[ $pkg =~ torch-geometric ]];
    then
        pip install ${pkg}
    else
        pip install ${pkg}+cpu -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
    fi
done

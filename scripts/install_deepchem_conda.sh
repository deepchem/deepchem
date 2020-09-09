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

if [ "$0" = "gpu" ];
then
    # We expect that the CUDA vesion is 10.1.
    # This is because TensorFlow mainly supports CUDA 10.1.
    cuda=cu101
    dgl_pkg=dgl-cu101
    echo "Installing DeepChem in the GPU environment"
else
    cuda=cpu
    dgl_pkg=dgl
    echo "Installing DeepChem in the CPU environment"
fi

# Install dependencies except PyTorch and TensorFlow
conda create -y --name deepchem python=$python_version
conda activate deepchem
conda env update --file $PWD/requirements.yml
pip install -r $PWD/requirements-test.txt

# Fixed packages
tensorflow=2.2.0
tensorflow_probability==0.10.1
torch=1.6.0
torchvision=0.7.0
pyg_torch=1.6.0

# Install TensorFlow dependencies
pip install tensorflow==$tensorflow tensorflow-probability==$tensorflow_probability

# Install PyTorch dependencies
if [ "$(uname)" == 'Darwin' ];
then
    # For MacOSX
    pip install torch==$torch torchvision==$torchvision
else
    pip install torch==$torch+$cuda torchvision==$torchvision+$cuda -f https://download.pytorch.org/whl/torch_stable.html
fi

# Install PyTorch Geometric and DGL dependencies
pip install torch-scatter==latest+$cuda -f https://pytorch-geometric.com/whl/torch-$pyg_torch.html
pip install torch-sparse==latest+$cuda -f https://pytorch-geometric.com/whl/torch-$pyg_torch.html
pip install torch-cluster==latest+$cuda -f https://pytorch-geometric.com/whl/torch-$pyg_torch.html
pip install torch-spline-conv==latest+$cuda -f https://pytorch-geometric.com/whl/torch-$pyg_torch.html
pip install torch-geometric
pip install $dgl_pkg
# install transformers package
pip install transformers

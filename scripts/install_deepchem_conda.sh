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
    cuda=cu101
    dgl_pkg=dgl-cu101
    echo "Installing DeepChem in the GPU envirionment"
else
    cuda=cpu
    dgl_pkg=dgl
    echo "Installing DeepChem in the CPU envirionment"
fi

# Install dependencies except PyTorch and TensorFlow
conda create -y --name deepchem python=$python_version
conda activate deepchem
conda env update --file $PWD/requirements.yml
pip install -r $PWD/requirements-test.txt

# Fixed packages
tensorflow=2.2.0
tensorflow_probability=0.10.1
torch=1.5.1
torchvision=0.6.1
torch_scatter=2.0.5
torch_sparse=0.6.6
torch_cluster=1.5.6
torch_spline_conv=1.2.0
torch_geometric=1.6.0
dgl=0.4.3.post2

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
TORCH=1.5.0
pip install torch-scatter==$torch_scatter+$cuda -f https://pytorch-geometric.com/whl/torch-$TORCH.html
pip install torch-sparse==$torch_sparse+$cuda -f https://pytorch-geometric.com/whl/torch-$TORCH.html
pip install torch-cluster==$torch_cluster+$cuda -f https://pytorch-geometric.com/whl/torch-$TORCH.html
pip install torch-spline-conv==$torch_spline_conv+$cuda -f https://pytorch-geometric.com/whl/torch-$TORCH.html
pip install torch-geometric==$torch_geometric
pip install $dgl_pkg==$dgl

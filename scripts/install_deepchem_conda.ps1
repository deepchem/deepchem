if ($python_version)
{
    echo "Using python "$python_version". But recommended to use python 3.6."
}
else
{
    echo "Using python 3.6 by default"
    $python_version=3.6
}

if($args[0] -eq "gpu")
{
    $cuda="cu101"
    dgl_pkg="dgl-cu101"
    echo "Installing DeepChem in the GPU envirionment"
}
else
{
    $cuda="cpu"
    $dgl_pkg="dgl"
    echo "Installing DeepChem in the CPU envirionment"
}

# Install dependencies except PyTorch and TensorFlow
conda create -y --name deepchem python=$python_version
conda activate deepchem
$path = Join-Path $Pwd "requirements.yml"
conda env update --file $path
$path = Join-Path $Pwd "requirements-test.txt"
pip install -r $path

# Fixed packages
$tensorflow=2.2.0
$tensorflow_probability=0.10.1
$torch=1.6.0
$torchvision=0.7.0
$pyg_torch=1.6.0

# Install Tensorflow dependencies
pip install tensorflow==$tensorflow tensorflow-probability==$tensorflow_probability

# Install PyTorch dependencies
pip install torch==$torch+$cuda torchvision==$torchvision+$cuda -f https://download.pytorch.org/whl/torch_stable.html

# Install PyTorch Geometric and DGL dependencies
pip install torch-scatter==latest+$cuda -f https://pytorch-geometric.com/whl/torch-$pyg_torch.html
pip install torch-sparse==latest+$cuda -f https://pytorch-geometric.com/whl/torch-$pyg_torch.html
pip install torch-cluster==latest+$cuda -f https://pytorch-geometric.com/whl/torch-$pyg_torch.html
pip install torch-spline-conv==latest+$cuda -f https://pytorch-geometric.com/whl/torch-$pyg_torch.html
pip install torch-geometric
pip install $dgl_pkg
# install transformers package
pip install transformers


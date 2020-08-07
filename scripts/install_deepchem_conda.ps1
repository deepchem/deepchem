if ($python_version)
{
    echo "Using python "$python_version". But recommended to use python 3.6."
}
else
{
    echo "Using python 3.6 by default"
    $python_version=3.6
}

if($args.Length -eq 1)
{
    $envname = $args[0]
    conda create -y --name $envname python=$python_version
    conda activate $envname
}
else
{
    echo "Installing DeepChem in current env"
}

if($args[1] -eq "gpu")
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
$path = Join-Path $Pwd "requirements.yml"
conda env update --file $path
$path = Join-Path $Pwd "requirements-test.txt"
pip install -r $path

# Fixed packages
$tensorflow=2.2.0
$tensorflow_probability=0.10.1
$torch=1.5.1
$torchvision=0.6.1
$torch_scatter=2.0.5
$torch_sparse=0.6.6
$torch_cluster=1.5.6
$torch_spline_conv=1.2.0
$torch_geometric=1.6.0
$dgl=0.4.3.post2

# Install Tensorflow dependencies
pip install tensorflow==$tensorflow tensorflow-probability==$tensorflow_probability

# Install PyTorch dependencies
pip install torch==$torch+$cuda torchvision==$torchvision+$cuda -f https://download.pytorch.org/whl/torch_stable.html

# Install PyTorch Geometric and DGL dependencies
$TORCH=1.5.0
pip install torch-scatter==$torch_scatter+$cuda -f https://pytorch-geometric.com/whl/torch-$TORCH.html
pip install torch-sparse==$torch_sparse+$cuda -f https://pytorch-geometric.com/whl/torch-$TORCH.html
pip install torch-cluster==$torch_cluster+$cuda -f https://pytorch-geometric.com/whl/torch-$TORCH.html
pip install torch-spline-conv==$torch_spline_conv+$cuda -f https://pytorch-geometric.com/whl/torch-$TORCH.html
pip install torch-geometric==$torch_geometric
pip install $dgl_pkg==$dgl

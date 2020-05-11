param($gpu)
if ($gpu -eq 1)
{
    $tensorflow = "tensorflow-gpu"
    echo "Using Tensorflow (GPU MODE)."
}
elseif($gpu -eq 0)
{
    $tensorflow = "tensorflow"
    echo "Using Tensorflow (CPU MODE)."
}
else
{
    $tensorflow = "tensorflow"
    echo "Using Tensorflow (CPU MODE) by default."
}
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

conda install -y -q -c deepchem -c rdkit -c conda-forge -c omnia `
    mdtraj `
    pdbfixer `
    rdkit `
    joblib `
    scikit-learn `
    networkx `
    pillow `
    pandas `
    nose `
    nose-timer `
    flaky `
    zlib `
    requests `
    py-xgboost `
    simdna `
    setuptools `
    biopython `
    numpy

pip install --pre -U $tensorflow tensorflow-probability

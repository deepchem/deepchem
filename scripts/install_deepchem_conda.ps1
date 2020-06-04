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

pip install -U tensorflow tensorflow-probability

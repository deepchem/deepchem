cd C:\Users\vagrant

# Install Miniconda.

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe -UseBasicParsing -OutFile Miniconda3-latest-Windows-x86_64.exe
.\Miniconda3-latest-Windows-x86_64.exe /S /D=C:\Miniconda3 | Out-Null
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Miniconda3;C:\Miniconda3\Scripts;C:\Miniconda3\Library\bin", [EnvironmentVariableTarget]::User)
& "C:\Miniconda3\Scripts\conda.exe" config --set ssl_verify no

# Install dependencies with conda.

& "C:\Miniconda3\Scripts\conda.exe" install -y -q -c deepchem -c rdkit -c conda-forge git mdtraj pdbfixer rdkit joblib scikit-learn networkx pillow pandas nose nose-timer flaky zlib requests py-xgboost simdna setuptools biopython numpy

# Install TensorFlow.

& "C:\Miniconda3\Scripts\pip.exe" install -U tensorflow tensorflow-probability

# Install Visual Studio redistributable library.

wget https://aka.ms/vs/16/release/vc_redist.x64.exe -UseBasicParsing -OutFile VC_redist.x64.exe
.\VC_redist.x64.exe /install /quiet

# Checkout DeepChem.

& "C:\Miniconda3\Library\bin\git.exe" clone https://github.com/deepchem/deepchem.git

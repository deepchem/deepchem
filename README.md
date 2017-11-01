# DeepChem
[![Build Status](https://travis-ci.org/deepchem/deepchem.svg?branch=master)](https://travis-ci.org/deepchem/deepchem)
[![Coverage Status](https://coveralls.io/repos/github/deepchem/deepchem/badge.svg?branch=master)](https://coveralls.io/github/deepchem/deepchem?branch=master)

DeepChem aims to provide a high quality open-source toolchain that
democratizes the use of deep-learning in drug discovery, materials science, quantum chemistry, and biology.

### Table of contents:

* [Requirements](#requirements)
* [Installation](#installation)
    * [Conda Environment](#using-a-conda-environment)
    * [Direct from Source](#installing-dependencies-manually)
    * [Docker](#using-a-docker-image)
* [FAQ](#faq)
* [Getting Started](#getting-started)
    * [Input Formats](#input-formats)
    * [Data Featurization](#data-featurization)
    * [Performances](#performances)
* [Contributing to DeepChem](#contributing-to-deepchem)
    * [Code Style Guidelines](#code-style-guidelines)
    * [Documentation Style Guidelines](#documentation-style-guidelines)
    * [Gitter](#gitter)
* [DeepChem Publications](#deepchem-publications)
* [Corporate Supporters](#corporate-supporters)
    * [Schrödinger](#schrödinger)
    * [DeepCrystal](#deep-crystal)
* [Examples](/examples)
* [About Us](#about-us)

## Requirements
* [pandas](http://pandas.pydata.org/)
* [rdkit](http://www.rdkit.org/docs/Install.html)
* [boost](http://www.boost.org/)
* [joblib](https://pypi.python.org/pypi/joblib)
* [sklearn](https://github.com/scikit-learn/scikit-learn.git)
* [numpy](https://store.continuum.io/cshop/anaconda/)
* [six](https://pypi.python.org/pypi/six)
* [mdtraj](http://mdtraj.org/)
* [tensorflow](https://www.tensorflow.org/)

## Installation

Installation from source is the only currently supported format. ```deepchem``` currently supports both Python 2.7 and Python 3.5, but is not supported on any OS'es except 64 bit linux. Please make sure you follow the directions below precisely. While you may already have system versions of some of these packages, there is no guarantee that `deepchem` will work with alternate versions than those specified below.

Note that when using Ubuntu 16.04 server or similar environments, you may need to ensure libxrender is provided via e.g.:
```bash
sudo apt-get install -y libxrender-dev
```

### Using a conda environment
You can install deepchem in a new conda environment using the conda commands in scripts/install_deepchem_conda.sh

```bash
git clone https://github.com/deepchem/deepchem.git      # Clone deepchem source code from GitHub
cd deepchem
bash scripts/install_deepchem_conda.sh deepchem
source activate deepchem
conda install -c conda-forge tensorflow-gpu=1.3.0      # If you want GPU support
python setup.py install                                # Manual install
nosetests -a '!slow' -v deepchem --nologcapture        # Run tests
```
This creates a new conda environment `deepchem` and installs in it the dependencies that
are needed. To access it, use the `source activate deepchem` command.
Check [this link](https://conda.io/docs/using/envs.html) for more information about
the benefits and usage of conda environments. **Warning**: Segmentation faults can [still happen](https://github.com/deepchem/deepchem/pull/379#issuecomment-277013514)
via this installation procedure.

### Easy Install via Conda
```bash
conda install -c deepchem -c rdkit -c conda-forge -c omnia deepchem=1.3.0
```

### Installing Dependencies Manually

1. Download the **64-bit** Python 2.7 or Python 3.5 versions of Anaconda for linux [here](https://www.continuum.io/downloads#_unix).
   Follow the [installation instructions](http://docs.continuum.io/anaconda/install#linux-install)

2. `rdkit`
   ```bash
   conda install -c rdkit rdkit
   ```

3. `joblib`
   ```bash
   conda install joblib
   ```

4. `six`
   ```bash
   pip install six
   ```
5. `networkx`
   ```bash
   conda install -c anaconda networkx=1.11
   ```

6. `mdtraj`
   ```bash
   conda install -c omnia mdtraj
   ```

7. `pdbfixer`
   ```bash
   conda install -c omnia pdbfixer=1.4
   ```

8. `tensorflow`: Installing `tensorflow` on older versions of Linux (which
    have glibc < 2.17) can be very challenging. For these older Linux versions,
    contact your local sysadmin to work out a custom installation. If your
    version of Linux is recent, then the following command will work:
    ```
    pip install tensorflow-gpu==1.3.0
    ```

9. `deepchem`: Clone the `deepchem` github repo:
   ```bash
   git clone https://github.com/deepchem/deepchem.git
   ```
   `cd` into the `deepchem` directory and execute
   ```bash
   python setup.py install
   ```

10. To run test suite, install `nosetests`:
   ```bash
   pip install nose
   ```
   Make sure that the correct version of `nosetests` is active by running
   ```bash
   which nosetests
   ```
   You might need to uninstall a system install of `nosetests` if
   there is a conflict.

11. If installation has been successful, all tests in test suite should pass:
    ```bash
    nosetests -v deepchem --nologcapture
    ```
    Note that the full test-suite uses up a fair amount of memory.
    Try running tests for one submodule at a time if memory proves an issue.

### Using a Docker Image
For major releases we will create docker environments with everything pre-installed.
In order to get GPU support you will have to use the 
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker) plugin.
``` bash
# This will the download the latest stable deepchem docker image into your images
docker pull deepchemio/deepchem

# This will create a container out of our latest image with GPU support
nvidia-docker run -i -t deepchemio/deepchem

# You are now in a docker container whose python has deepchem installed
# For example you can run our tox21 benchmark
cd deepchem/examples
python benchmark.py -d tox21

# Or you can start playing with it in the command line
pip install jupyter
ipython
import deepchem as dc
```

## FAQ
1. Question: I'm seeing some failures in my test suite having to do with MKL
   ```Intel MKL FATAL ERROR: Cannot load libmkl_avx.so or libmkl_def.so.```

   Answer: This is a general issue with the newest version of `scikit-learn` enabling MKL by default. This doesn't play well with many linux systems. See BVLC/caffe#3884 for discussions. The following seems to fix the issue
   ```bash
   conda install nomkl numpy scipy scikit-learn numexpr
   conda remove mkl mkl-service
   ```

## Getting Started
The first step to getting started is looking at the examples in the `examples/` directory. Try running some of these examples on your system and verify that the models train successfully. Afterwards, to apply `deepchem` to a new problem, try starting from one of the existing examples and modifying it step by step to work with your new use-case.

### Input Formats
Accepted input formats for deepchem include csv, pkl.gz, and sdf files. For
example, with a csv input, in order to build models, we expect the
following columns to have entries for each row in the csv file.

1. A column containing SMILES strings [1].
2. A column containing an experimental measurement.
3. (Optional) A column containing a unique compound identifier.

Here's an example of a potential input file.

|Compound ID    | measured log solubility in mols per litre | smiles         |
|---------------|-------------------------------------------|----------------|
| benzothiazole | -1.5                                      | c2ccc1scnc1c2  |


Here the "smiles" column contains the SMILES string, the "measured log
solubility in mols per litre" contains the experimental measurement and
"Compound ID" contains the unique compound identifier.

[2] Anderson, Eric, Gilman D. Veith, and David Weininger. "SMILES, a line
notation and computerized interpreter for chemical structures." US
Environmental Protection Agency, Environmental Research Laboratory, 1987.

### Data Featurization

Most machine learning algorithms require that input data form vectors.
However, input data for drug-discovery datasets routinely come in the
format of lists of molecules and associated experimental readouts. To
transform lists of molecules into vectors, we need to subclasses of DeepChem
loader class ```dc.data.DataLoader``` such as ```dc.data.CSVLoader``` or
```dc.data.SDFLoader```. Users can subclass ```dc.data.DataLoader``` to
load arbitrary file formats. All loaders must be
passed a ```dc.feat.Featurizer``` object. DeepChem provides a number of
different subclasses of ```dc.feat.Featurizer``` for convenience.

### Performances
In depth performance tables for DeepChem models are available on [MoleculeNet.ai](https://moleculenet.ai)

### Gitter
Join us on gitter at [https://gitter.im/deepchem/Lobby](https://gitter.im/deepchem/Lobby). Probably the easiest place to ask simple questions or float requests for new features.

## DeepChem Publications
1. [Computational Modeling of β-secretase 1 (BACE-1) Inhibitors using
Ligand Based
Approaches](http://pubs.acs.org/doi/abs/10.1021/acs.jcim.6b00290)
2. [Low Data Drug Discovery with One-Shot Learning](http://pubs.acs.org/doi/abs/10.1021/acscentsci.6b00367)
3. [MoleculeNet: A Benchmark for Molecular Machine Learning](https://arxiv.org/abs/1703.00564)
4. [Atomic Convolutional Networks for Predicting Protein-Ligand Binding Affinity](https://arxiv.org/abs/1703.10603)

## About Us
DeepChem is possible due to notable contributions from many people including Peter Eastman, Evan Feinberg, Joe Gomes, Karl Leswing, Vijay Pande, Aneesh Pappu, Bharath Ramsundar and Michael Wu (alphabetical ordering).  DeepChem was originally created by [Bharath Ramsundar](http://rbharath.github.io/) with encouragement and guidance from [Vijay Pande](https://pande.stanford.edu/).

DeepChem started as a [Pande group](https://pande.stanford.edu/) project at Stanford, and is now developed by many academic and industrial collaborators. DeepChem actively encourages new academic and industrial groups to contribute!

## Corporate Supporters
DeepChem is supported by a number of corporate partners who use DeepChem to solve interesting problems.

### Schrödinger
[![Schödinger](https://github.com/deepchem/deepchem/blob/master/docs/source/_static/schrodinger_logo.png)](https://www.schrodinger.com/)

> DeepChem has transformed how we think about building QSAR and QSPR models when very large data sets are available; and we are actively using DeepChem to investigate how to best combine the power of deep learning with next generation physics-based scoring methods.

### DeepCrystal
<img src="https://github.com/deepchem/deepchem/blob/master/docs/source/_static/deep_crystal_logo.png" alt="DeepCrystal Logo" height=150px/>

> DeepCrystal was an early adopter of DeepChem, which we now rely on to abstract away some of the hardest pieces of deep learning in drug discovery. By open sourcing these efficient implementations of chemically / biologically aware deep-learning systems, DeepChem puts the latest research into the hands of the scientists that need it, materially pushing forward the field of in-silico drug discovery in the process.


## Version
1.2.0

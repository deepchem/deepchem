# DeepChem

[![Anaconda-Server Badge](https://anaconda.org/conda-forge/deepchem/badges/version.svg)](https://anaconda.org/conda-forge/deepchem)
[![PyPI version](https://badge.fury.io/py/deepchem.svg)](https://pypi.org/project/deepchem/)
[![Documentation Status](https://readthedocs.org/projects/deepchem/badge/?version=latest)](https://deepchem.readthedocs.io/en/latest/?badge=latest) 
[![Test for DeepChem Core](https://github.com/deepchem/deepchem/workflows/Test%20for%20DeepChem%20Core/badge.svg)](https://github.com/deepchem/deepchem/actions?query=workflow%3A%22Test+for+DeepChem+Core%22)
[![Test for documents](https://github.com/deepchem/deepchem/workflows/Test%20for%20documents/badge.svg)](https://github.com/deepchem/deepchem/actions?query=workflow%3A%22Test+for+documents%22)
[![Test for build scripts](https://github.com/deepchem/deepchem/workflows/Test%20for%20build%20scripts/badge.svg)](https://github.com/deepchem/deepchem/actions?query=workflow%3A%22Test+for+build+scripts%22)
[![codecov](https://codecov.io/gh/deepchem/deepchem/branch/master/graph/badge.svg?token=5rOZB2BY3h)](https://codecov.io/gh/deepchem/deepchem)

[Website](https://deepchem.io/) | [Documentation](https://deepchem.readthedocs.io/en/latest/) | [Colab Tutorial](https://github.com/deepchem/deepchem/tree/master/examples/tutorials) | [Discussion Forum](https://forum.deepchem.io/) | [Discord](https://discord.gg/cGzwCdrUqS) | [Model Wishlist](https://github.com/deepchem/deepchem/issues/2680) | [Tutorial Wishlist](https://github.com/deepchem/deepchem/issues/2907)

DeepChem aims to provide a high-quality open-source toolchain
that democratizes the use of deep learning in drug discovery,
materials science, quantum chemistry, and biology.

### Table of contents:

- [Quickstart](#quickstart)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Stable version](#stable-version)
  - [Nightly build version](#nightly-build-version)
  - [Docker](#docker)`
  - [From source](#from-source)
- [Getting Started](#getting-started)
  - [Discord](#discord)
- [About Us](#about-us)
- [Contributing to DeepChem](/CONTRIBUTING.md)
- [Citing DeepChem](#citing-deepchem)

## Quickstart

Train a machine learning model on molecular data in just a few lines of code:

```python
import deepchem as dc
from sklearn.ensemble import RandomForestRegressor

# 1. Load Delaney (ESOL) molecular solubility dataset from MoleculeNet
tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='ECFP')
train_dataset, valid_dataset, test_dataset = datasets

# 2. Fit a machine learning model
rf = RandomForestRegressor(n_estimators=100)
model = dc.models.SklearnModel(model=rf)
model.fit(train_dataset)

# 3. Evaluate the model on test data
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
test_scores = model.evaluate(test_dataset, [metric], transformers)
print(f"Test Pearson R^2 Score: {test_scores['pearson_r2_score']:.3f}")
```

DeepChem also provides native deep learning models built on PyTorch, TensorFlow, and JAX:

```python
# Train a Graph Convolutional Neural Network on molecular graphs
tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = datasets

model = dc.models.GraphConvModel(n_tasks=len(tasks), mode='regression')
model.fit(train_dataset, nb_epoch=30)
```

## Requirements

DeepChem currently supports Python 3.8 through 3.11 and requires these core packages:

- [joblib](https://pypi.python.org/pypi/joblib)
- [NumPy](https://numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [SciPy](https://www.scipy.org/)
- [rdkit](https://www.rdkit.org/)
- [SymPy](https://www.sympy.org/)

### Soft Requirements

DeepChem has a number of "soft" requirements for specialized models and frameworks.
If you encounter errors such as `ImportError: This class requires XXXX`, you may need to install optional packages.

Please check [the requirements guide](https://deepchem.readthedocs.io/en/latest/get_started/requirements.html#soft-requirements) for details on soft requirements.

## Installation

### Stable version

The stable version of DeepChem can be installed using pip or conda:

```bash
pip install deepchem
```
or
```bash
conda install -c conda-forge deepchem
```

DeepChem provides support for deep learning frameworks and specialized toolkits via optional pip extras:

For models with TensorFlow dependencies:
```bash
pip install deepchem[tensorflow]
```
For models with PyTorch dependencies:
```bash
pip install deepchem[torch]
```
For models with JAX dependencies:
```bash
pip install deepchem[jax]
```
For Density Functional Theory / Quantum Chemistry dependencies:
```bash
pip install deepchem[dqc]
```

If GPU support is required, make sure CUDA is installed and install the desired deep learning framework matching your CUDA version before installing DeepChem:

1. tensorflow - https://www.tensorflow.org/install
2. pytorch - https://pytorch.org/get-started/locally/#start-locally
3. jax - https://github.com/google/jax#pip-installation-gpu-cuda

In `zsh`, square brackets are used for globbing/pattern matching. Escape them or wrap the package in quotes: `pip install 'deepchem[torch]'`.

### Nightly build version
The nightly version is built from the HEAD of DeepChem. It can be installed using:

```bash
pip install --pre deepchem
```

### Docker

If you want to use DeepChem with Docker, you can pull prebuilt images from DockerHub:
DockerHub : https://hub.docker.com/repository/docker/deepchemio/deepchem

- `deepchemio/deepchem:x.x.x`
  - Image built using Conda (x.x.x is a version of DeepChem)
  - The x.x.x image is built when a version tag is pushed
  - Dockerfile is located in the `docker/tag` directory
- `deepchemio/deepchem:latest`
  - Image built from source code
  - The latest image is built every time changes are merged into the master branch
  - Dockerfile is located in the `docker/nightly` directory

You can pull the latest image:

```bash
docker pull deepchemio/deepchem:latest
```

For GPU-enabled containers:

```bash
docker run --gpus all --rm -it deepchemio/deepchem:latest
```

If you want to know Docker usages with DeepChem in more detail, please check [the Docker tutorial](https://deepchem.readthedocs.io/en/latest/get_started/installation.html#docker).

### From source

If you want to install all soft dependencies at once or contribute to DeepChem, we recommend installing DeepChem from source.

Please check [the source installation guide](https://deepchem.readthedocs.io/en/latest/get_started/installation.html#from-source-with-conda).

## Getting Started

The DeepChem project maintains an extensive collection of [tutorials](https://github.com/deepchem/deepchem/tree/master/examples/tutorials). All tutorials are designed to be run on Google colab (or locally if you prefer). Tutorials are arranged in a suggested learning sequence which will take you from beginner to proficient at molecular machine learning and computational biology more broadly.

After working through the tutorials, you can also go through other [examples](https://github.com/deepchem/deepchem/tree/master/examples). To apply `deepchem` to a new problem, try starting from one of the existing examples or tutorials and modifying it step by step to work with your new use-case. If you have questions or comments you can raise them on our [gitter](https://gitter.im/deepchem/Lobby).

### Supported Integrations

- [Weights & Biases](https://docs.wandb.ai/guides/integrations/other/deepchem): Track your DeepChem model's training and evaluation metrics.

### Discord

The DeepChem [Discord](https://discord.gg/cGzwCdrUqS) hosts a number of scientists, developers, and enthusiasts interested in deep learning for the life sciences. Probably the easiest place to ask simple questions or float requests for new features.

## About Us

DeepChem is managed by a team of open source contributors. Anyone is free to join and contribute!

## Citing DeepChem

If you have used DeepChem in the course of your research, we ask that you cite the "Deep Learning for the Life Sciences" book by the DeepChem core team.

To cite this book, please use this bibtex entry:

```
@book{Ramsundar-et-al-2019,
    title={Deep Learning for the Life Sciences},
    author={Bharath Ramsundar and Peter Eastman and Patrick Walters and Vijay Pande and Karl Leswing and Zhenqin Wu},
    publisher={O'Reilly Media},
    note={\url{https://www.amazon.com/Deep-Learning-Life-Sciences-Microscopy/dp/1492039837}},
    year={2019}
}
```

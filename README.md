# DeepChem

[![Build Status](https://travis-ci.org/deepchem/deepchem.svg?branch=master)](https://travis-ci.org/deepchem/deepchem)
[![Coverage Status](https://coveralls.io/repos/github/deepchem/deepchem/badge.svg?branch=master)](https://coveralls.io/github/deepchem/deepchem?branch=master)
[![Documentation Status](https://readthedocs.org/projects/deepchem/badge/?version=latest)](https://deepchem.readthedocs.io/en/latest/?badge=latest)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/deepchem/badges/version.svg)](https://anaconda.org/conda-forge/deepchem)
[![PyPI version](https://badge.fury.io/py/deepchem.svg)](https://badge.fury.io/py/deepchem)

[Website](https://deepchem.io/) | [Documentation](https://deepchem.readthedocs.io/en/latest/) | [Colab Tutorial](https://github.com/deepchem/deepchem/tree/master/examples/tutorials) | [Discussion Forum](https://forum.deepchem.io/) | [Gitter](https://gitter.im/deepchem/Lobby)

DeepChem aims to provide a high quality open-source toolchain
that democratizes the use of deep-learning in drug discovery,
materials science, quantum chemistry, and biology.

### Table of contents:

- [Requirements](#requirements)
- [Installation](#installation)
  - [Stable version](#stable-version)
  - [Nightly build version](#nightly-build-version)
  - [Docker](#docker)
  - [From source](#from-source)
- [Getting Started](#getting-started)
- [Contributing to DeepChem](/CONTRIBUTING.md)
  - [Code Style Guidelines](/CONTRIBUTING.md#code-style-guidelines)
  - [Documentation Style Guidelines](/CONTRIBUTING.md#documentation-style-guidelines)
  - [Gitter](#gitter)
- [About Us](#about-us)
- [Citing DeepChem](#citing-deepchem)

## Requirements

DeepChem currently supports Python 3.5 through 3.7 and requires these packages on any condition.

- [joblib](https://pypi.python.org/pypi/joblib)
- [NumPy](https://numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [SciPy](https://www.scipy.org/)
- [TensorFlow](https://www.tensorflow.org/)
  - `deepchem>=2.4.0` requires tensorflow v2
  - `deepchem<2.4.0` requires tensorflow v1

### Soft Requirements

DeepChem has a number of "soft" requirements.  
If you face some errors like `ImportError: No module named XXXX`, you may need to install some packages.

Please check [the document](https://deepchem.readthedocs.io/en/latest/requirements.html##soft-requirements) about soft requirements.

## Installation

### Stable version

**Caution!! : The latest stable version was published nearly a year ago. If you are a pip user or you face some errors, we recommend the nightly build version.**

RDKit is a soft requirement package, but many useful methods like molnet depend on it. We recommend installing RDKit with deepchem.

```bash
pip install tensorflow==1.14
conda install -y -c conda-forge rdkit deepchem==2.3.0
```

If you want GPU support:

```bash
pip install tensorflow-gpu==1.14
conda install -y -c conda-forge rdkit deepchem==2.3.0
```

### Nightly build version

You install the nightly build version via pip. The nightly version is built by the HEAD of DeepChem.

```bash
pip install tensorflow==2.3.0
pip install --pre deepchem
```

RDKit is a soft requirement package, but many useful methods like molnet depend on it. We recommend installing RDKit with deepchem if you use conda.

```bash
conda install -y -c conda-forge rdkit
```

### Docker

If you want to install deepchem using a docker, you can pull two kinds of images.  
DockerHub : https://hub.docker.com/repository/docker/deepchemio/deepchem

- `deepchemio/deepchem:x.x.x`
  - Image built by using a conda package manager (x.x.x is a version of deepchem)
  - The x.x.x image is built when we push x.x.x. tag
  - Dockerfile is put in `docker/conda-forge` directory
- `deepchemio/deepchem:latest`
  - Image built by the master branch of deepchem source codes
  - The latest image is built every time we commit to the master branch
  - Dockerfile is put in `docker/master` directory

You pull the image like this.

```bash
docker pull deepchemio/deepchem:2.3.0
```

If you want to know docker usages with deepchem in more detail, please check [the document](https://deepchem.readthedocs.io/en/latest/installation.html#docker).

### From source

If you try install all soft dependencies at once or contribute to deepchem, we recommend you should install deepchem from source.

Please check [this introduction](https://deepchem.readthedocs.io/en/latest/installation.html#from-source).

## Getting Started

The DeepChem project maintains an extensive collection of [tutorials](https://github.com/deepchem/deepchem/tree/master/examples/tutorials). All tutorials are designed to be run on Google colab (or locally if you prefer). Tutorials are arranged in a suggested learning sequence which will take you from beginner to proficient at molecular machine learning and computational biology more broadly.

After working through the tutorials, you can also go through other [examples](https://github.com/deepchem/deepchem/tree/master/examples). To apply `deepchem` to a new problem, try starting from one of the existing examples or tutorials and modifying it step by step to work with your new use-case. If you have questions or comments you can raise them on our [gitter](https://gitter.im/deepchem/Lobby).

### Gitter

Join us on gitter at [https://gitter.im/deepchem/Lobby](https://gitter.im/deepchem/Lobby). Probably the easiest place to ask simple questions or float requests for new features.

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

## Version

2.4.0-rc

# Requirements
During its installation, DeepChem requires various prerequisite libraries to be installed. The various `.yml` files in the `requirements` folder contain the different libraries that are the prerequisites based on the installer, operating system, etc. The different folders are used to show the dependencies of the DeepChem library on the respective deep-learning backends.

The files present are:
1. `env_common.yml` - contains libraries related to scientific applications that are common to all operating systems.
2. `env_dqc.yml` - contains libraries specific for using Quantum Chemistry packages through the [Differentiable Quantum Chemistry package](https://github.com/diffqc/dqc).
3. `env_mac.yml` - contains dependencies specific to macOS.
4. `env_ubuntu.yml` - contains dependencies specific to Ubuntu.
5. `env_test.yml` - contains libraries related to testing code and models.

The sub-folders present are for JAX, TensorFlow and PyTorch. The sub-folder contain the cpu and gpu deep learning backend requirements of DeepChem.

## env_common.yml
This file contains the bare minimum requirements to run the DeepChem library. It is not specific to any operating system.
1. `openmm` - It is a [toolkit](https://github.com/openmm/openmm) for molecular simulation using high performance GPU code. 
2. `mdtraj` - It is an [open library](https://github.com/mdtraj/mdtraj) for the analysis of molecular dynamics trajectories.
3. `pdbfixer` - It is an easy to use [application](https://github.com/openmm/pdbfixer) for fixing problems in Protein Data Bank files in preparation for simulating them.
4. `pip` - It is an installer [package](https://github.com/pypa/pip) for Python.

Using the `pip` installer package in Python, we can download various other requirements for the deepchem library:
1. `numpy` - [NumPy](https://github.com/numpy/numpy) is the fundamental package for scientific computing with Python.
2. `rdkit` - [RDKit](https://github.com/rdkit/rdkit) is a collection of cheminformatics and machine-learning software written in C++ and Python.
3. `pre-commit` - A [framework](https://github.com/pre-commit/pre-commit) for managing and maintaining multi-language pre-commit hooks. 
4. `biopython` - The [Biopython](https://github.com/biopython/biopython) Project is an international association of developers of freely available Python tools for computational molecular biology.
5. `dgllife` - [DGL-LifeSci](https://github.com/awslabs/dgl-lifesci) is a DGL-based package for various applications in life science with graph neural networks. DeepChem works on the `0.2.8` version.
6. `lightgbm` - [LightGBM](https://github.com/microsoft/LightGBM) is a gradient boosting framework that uses tree based learning algorithm. DeepChem works on any `3.x` version.
7. `matminer` - [matminer](https://github.com/hackingmaterials/matminer) is a library for performing data mining in the field of materials science.
8. `mordred` - It is a molecular descriptor calculator [package](https://github.com/mordred-descriptor/mordred).
9. `networkx` - [NetworkX](https://github.com/networkx/networkx) is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.
10. `pillow` - It is the friendly [fork](https://github.com/python-pillow/Pillow) of the Python Imaging Library(PIL).
11. `pubchempy` - It is a Python [wrapper](https://github.com/mcs07/PubChemPy) for the PubChem PUG REST API.
12. `pyGPGO` - [pyGPGO](https://github.com/josejimenezluna/pyGPGO) is a simple and modular Python (>3.5) package for bayesian optimization.
13. `pymatgen` - Python Materials Genomics (pymatgen) is a robust materials analysis [code](https://github.com/materialsproject/pymatgen) that defines classes for structures and molecules with support for many electronic structure codes.
14. `simdna` - A Python [library](https://github.com/kundajelab/simdna) for creating simulated regulatory DNA sequences.
15. `transformers` - [Transformers](https://github.com/huggingface/transformers) provides thousands of pretrained models to perform tasks on different modalities such as text, vision, and audio. DeepChem works on any `4.10.x` version.
16. `xgboost` - [XGBoost](https://github.com/dmlc/xgboost) is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable.
17. `gensim` - [Gensim](https://github.com/RaRe-Technologies/gensim) is a Python library for topic modelling, document indexing and similarity retrieval with large corpora. It's used by the `mol2vec` library.
18. `tensorboard` - [TensorBoard](https://github.com/tensorflow/tensorboard) is a suite of web applications for inspecting and understanding your TensorFlow runs and graphs.

## env_dqc.yml
This file contains the requirements for using the [Differentiable Quantum Chemistry package](https://github.com/diffqc/dqc). Using `pip`, we can install the following packages:
1. `numpy` -  [NumPy](https://github.com/numpy/numpy) is the fundamental package for scientific computing with Python.
2. `scipy` - [SciPy](https://github.com/scipy/scipy) is an open-source software for mathematics, science, and engineering.
3. `h5py` - HDF5 for Python - The [h5py package](https://github.com/h5py/h5py) is a Pythonic interface to the HDF5 binary data format. 
4. `PyTorch` - Download the suitable version of [PyTorch](https://pytorch.org/) based on your computer. The version of `torch` should be `1.12.0+cpu`.
5. `DQC` - Download the [Differentiable Quantum Chemistry package](https://github.com/diffqc/dqc).
6. `xitorch` - [xitorch](https://github.com/xitorch/xitorch) is a PyTorch-based library of differentiable functions and functionals that can be widely used in scientific computing applications as well as deep learning.
7. `pylibxc2` - A thin [wrapper](https://github.com/mfkasim1/pylibxc/) of libxc. 

## env_mac.yml
This file contains the requirements specific to macOS. The dependencies are as follows:
1. `hhsuite` - The [HH-suite](https://github.com/soedinglab/hh-suite) is an open-source software package for sensitive protein sequence searching based on the pairwise alignment of hidden Markov models (HMMs).
2. `vina` - [AutoDock Vina](https://github.com/ccsb-scripps/AutoDock-Vina) is one of the fastest and most widely used open-source docking engines. It can be installed with `pip`.

## env_ubuntu.yml
This file contains the requirements specific to the Ubuntu distro of the Linux Operating System. The dependency is as follows:
1. `hhsuite` - The [HH-suite](https://github.com/soedinglab/hh-suite) is an open-source software package for sensitive protein sequence searching based on the pairwise alignment of hidden Markov models (HMMs).

## env_test.yml
This file contains the requiremetns to perform testing of models and code in the DeepChem library. The dependencies can be installed using `pip` and are as follows:
1. `flake8` - [flake8](https://github.com/PyCQA/flake8) is a python tool that glues together pycodestyle, pyflakes, and mccabe.
2. `flaky` - [Flaky](https://github.com/box/flaky) is a plugin for nose or pytest that automatically reruns flaky tests.
3. `mypy` - [Mypy](https://github.com/python/mypy) is a static type checker for Python.
4. `pytest` - The [pytest framework](https://github.com/pytest-dev/pytest) makes it easy to write small tests.
5. `pytest-cov` - It's a coverage [plugin](https://github.com/pytest-dev/pytest-cov) for `pytest`.
6. `types-pkg_resources` - This is an auto-generated PEP 561 type stub package for pkg\_resources package. It can be used by type-checking tools like mypy, PyCharm, pytype etc. to check code that uses pkg\_resources. Its source is [here](https://github.com/python/typeshed/tree/master/stubs/pkg_resources).
7. `types-setuptools` - This is a PEP 561 type stub package for the setuptools package. It can be used by type-checking tools like mypy, pyright, pytype, PyCharm, etc. to check code that uses setuptools. Its source is [here](https://github.com/python/typeshed/tree/main/stubs/setuptools).
8. `yapf` - It is a [formatter](https://github.com/google/yapf) for Python files.

## jax
### env_jax.cpu.yml
This file specifies the dependencies that JAX uses for running on the CPU.

### env_jax.gpu.yml
This file specifies the dependencies that JAX uses for running on the GPU.

## tensorflow

### env_tensorflow.cpu.yml
This file contains the dependencies that TensorFlow uses for running on the CPU.


## torch

### env_torch.cpu.yml
This file contains the dependencies that PyTorch uses for running on the CPU.

### env_torch.gpu.yml
This file contains the dependencies that PyTorch uses for running on the GPU.

### env_torch.mac.cpu.yml
This file contains the dependencies that PyTorch uses for running on the CPU for macOS.


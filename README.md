# DeepChem
[![Build Status](https://travis-ci.org/deepchem/deepchem.svg?branch=master)](https://travis-ci.org/deepchem/deepchem)
[![Coverage Status](https://coveralls.io/repos/github/deepchem/deepchem/badge.svg?branch=master)](https://coveralls.io/github/deepchem/deepchem?branch=master)

DeepChem aims to provide a high quality open-source toolchain that
democratizes the use of deep-learning in drug discovery, materials science, and quantum
chemistry. DeepChem is a package developed by the [Pande group](https://pande.stanford.edu/) at
Stanford and originally created by [Bharath Ramsundar](http://rbharath.github.io/).

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

### Using a conda environment
You can install deepchem in a new conda environment using the conda commands in scripts/install_deepchem_conda.sh

```bash
git clone https://github.com/deepchem/deepchem.git      # Clone deepchem source code from GitHub
cd deepchem
bash scripts/install_deepchem_conda.sh deepchem
source activate deepchem
pip install tensorflow-gpu==1.0.1                       # If you want GPU support
python setup.py install                                 # Manual install
nosetests -v deepchem --nologcapture                    # Run tests
```
This creates a new conda environment `deepchem` and installs in it the dependencies that
are needed. To access it, use the `source activate deepchem` command.
Check [this link](https://conda.io/docs/using/envs.html) for more information about
the benefits and usage of conda environments. **Warning**: Segmentation faults can [still happen](https://github.com/deepchem/deepchem/pull/379#issuecomment-277013514)
via this installation procedure.

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
    pip install tensorflow-gpu
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
For major releases we will create docker environments with everything pre-installed
``` bash
# This will the download the latest stable deepchem docker image into your images
docker pull deepchemio/deepchem

# This will create a container out of our latest image
docker run -i -t deepchemio/deepchem

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
* Classification

Index splitting

|Dataset    |Model               |Train score/ROC-AUC|Valid score/ROC-AUC|
|-----------|--------------------|-------------------|-------------------|
|clintox    |Logistic regression |0.967              |0.676              |
|           |Random forest       |0.995              |0.776              |
|           |XGBoost             |0.879              |0.890              |
|           |IRV                 |0.763              |0.814              |
|           |MT-NN classification|0.934              |0.830              |
|           |Robust MT-NN        |0.949              |0.827              |
|           |Graph convolution   |0.946              |0.860              |
|           |Weave               |0.942              |0.917              |
|hiv        |Logistic regression |0.864              |0.739              |
|           |Random forest       |0.999              |0.720              |
|           |XGBoost             |0.917              |0.745              |
|           |IRV                 |0.841              |0.724              |
|           |NN classification   |0.761              |0.652              |
|           |Robust NN           |0.780              |0.708              |
|           |Graph convolution   |0.876              |0.779              |
|           |Weave               |0.907              |0.753              |
|muv        |Logistic regression |0.963              |0.766              |
|           |XGBoost             |0.895              |0.714              |
|           |MT-NN classification|0.904              |0.764              |
|           |Robust MT-NN        |0.934              |0.781              |
|           |Graph convolution   |0.840              |0.823              |
|           |Weave               |0.762              |0.761              |
|pcba       |Logistic regression |0.809              |0.776              |
|           |XGBoost             |0.931              |0.847              |
|           |MT-NN classification|0.826              |0.802              |
|           |Robust MT-NN        |0.809              |0.783              |
|           |Graph convolution   |0.876              |0.852              |
|sider      |Logistic regression |0.933              |0.620              |
|           |Random forest       |0.999              |0.670              |
|           |XGBoost             |0.829              |0.639              |
|           |IRV                 |0.649              |0.642              |
|           |MT-NN classification|0.775              |0.634              |
|           |Robust MT-NN        |0.803              |0.632              |
|           |Graph convolution   |0.708              |0.594              |
|           |Weave               |0.591              |0.580              |
|tox21      |Logistic regression |0.903              |0.705              |
|           |Random forest       |0.999              |0.733              |
|           |XGBoost             |0.891              |0.753              |
|           |IRV                 |0.811              |0.767              |
|           |MT-NN classification|0.856              |0.763              |
|           |Robust MT-NN        |0.857              |0.767              |
|           |Graph convolution   |0.872              |0.798              |
|           |Weave               |0.810              |0.778              |
|toxcast    |Logistic regression |0.721              |0.575              |
|           |XGBoost             |0.738              |0.621              |
|           |MT-NN classification|0.830              |0.678              |
|           |Robust MT-NN        |0.825              |0.680              |
|           |Graph convolution   |0.821              |0.720              |
|           |Weave               |0.766              |0.715              |

Random splitting

|Dataset    |Model               |Train score/ROC-AUC|Valid score/ROC-AUC|
|-----------|--------------------|-------------------|-------------------|
|bace_c     |Logistic regression |0.954              |0.850              |
|           |Random forest       |0.999              |0.939              |
|           |IRV                 |0.876              |0.871              |
|           |NN classification   |0.877              |0.790              |
|           |Robust NN           |0.887              |0.864              |
|           |Graph convolution   |0.906              |0.861              |
|           |Weave               |0.807              |0.780              |
|bbbp       |Logistic regression |0.980              |0.876              |
|           |Random forest       |0.999              |0.918              |
|           |IRV                 |0.904              |0.917              |
|           |NN classification   |0.882              |0.915              |
|           |Robust NN           |0.878              |0.878              |
|           |Graph convolution   |0.962              |0.897              |
|           |Weave               |0.929              |0.934              |
|clintox    |Logistic regression |0.972              |0.725              |
|           |Random forest       |0.997              |0.670              |
|           |XGBoost             |0.886              |0.731              |
|           |IRV                 |0.809              |0.846              |
|           |MT-NN classification|0.951              |0.834              |
|           |Robust MT-NN        |0.959              |0.830              |
|           |Graph convolution   |0.975              |0.876              |
|           |Weave               |0.945              |0.818              |
|hiv        |Logistic regression |0.860              |0.806              |
|           |Random forest       |0.999              |0.850              |
|           |XGBoost             |0.933              |0.841              |
|           |IRV                 |0.839              |0.809              |
|           |NN classification   |0.742              |0.715              |
|           |Robust NN           |0.753              |0.727              |
|           |Graph convolution   |0.847              |0.803              |
|           |Weave               |0.902              |0.825              |
|muv        |Logistic regression |0.957              |0.719              |
|           |XGBoost             |0.874              |0.696              |
|           |MT-NN classification|0.902              |0.734              |
|           |Robust MT-NN        |0.933              |0.732              |
|           |Graph convolution   |0.860              |0.730              |
|           |Weave               |0.763              |0.763              |
|pcba       |Logistic regression |0.808        	     |0.776              |
|           |MT-NN classification|0.811        	     |0.778              |
|           |Robust MT-NN        |0.811              |0.771              |
|           |Graph convolution   |0.872       	     |0.844              |
|sider      |Logistic regression |0.929        	     |0.656              |
|           |Random forest       |0.999              |0.665              |
|           |XGBoost             |0.824              |0.635              |
|           |IRV                 |0.648              |0.596              |
|           |MT-NN classification|0.777        	     |0.655              |
|           |Robust MT-NN        |0.804              |0.630              |
|           |Graph convolution   |0.705        	     |0.618              |
|           |Weave               |0.616              |0.645              |
|tox21      |Logistic regression |0.902              |0.715              |
|           |Random forest       |0.999              |0.764              |
|           |XGBoost             |0.874              |0.773              |
|           |IRV                 |0.808              |0.767              |
|           |MT-NN classification|0.844              |0.795              |
|           |Robust MT-NN        |0.855              |0.773              |
|           |Graph convolution   |0.865              |0.827              |
|           |Weave               |0.837              |0.830              |
|toxcast    |Logistic regression |0.725        	     |0.586              |
|           |XGBoost             |0.738              |0.633              |
|           |MT-NN classification|0.836        	     |0.684              |
|           |Robust MT-NN        |0.822              |0.681              |
|           |Graph convolution   |0.820        	     |0.717              |
|           |Weave               |0.757              |0.729              |

Scaffold splitting

|Dataset    |Model               |Train score/ROC-AUC|Valid score/ROC-AUC|
|-----------|--------------------|-------------------|-------------------|
|bace_c     |Logistic regression |0.957              |0.729              |
|           |Random forest       |0.999              |0.720              |
|           |IRV                 |0.899              |0.701              |
|           |NN classification   |0.897              |0.743              |
|           |Robust NN           |0.910              |0.747              |
|           |Graph convolution   |0.920              |0.682              |
|           |Weave               |0.860              |0.629              |
|bbbp       |Logistic regression |0.980              |0.959              |
|           |Random forest       |0.999              |0.953              |
|           |IRV                 |0.914              |0.961              |
|           |NN classification   |0.899              |0.961              |
|           |Robust NN           |0.908              |0.956              |
|           |Graph convolution   |0.968              |0.950              |
|           |Weave               |0.925              |0.968              |
|clintox    |Logistic regression |0.965              |0.688              |
|           |Random forest       |0.993              |0.735              |
|           |XGBoost             |0.873              |0.850              |
|           |IRV                 |0.793              |0.718              |
|           |MT-NN classification|0.937              |0.828              |
|           |Robust MT-NN        |0.956              |0.821              |
|           |Graph convolution   |0.965              |0.900              |
|           |Weave               |0.950              |0.947              |
|hiv        |Logistic regression |0.858              |0.798              |
|           |Random forest       |0.946              |0.562              |
|           |XGBoost             |0.927              |0.830              |
|           |IRV                 |0.847              |0.811              |
|           |NN classification   |0.775              |0.765              |
|           |Robust NN           |0.785              |0.748              |
|           |Graph convolution   |0.867              |0.769              |
|           |Weave               |0.875              |0.816              |
|muv        |Logistic regression |0.947              |0.767              |
|           |XGBoost             |0.875              |0.705              |
|           |MT-NN classification|0.899              |0.762              |
|           |Robust MT-NN        |0.944              |0.726              |
|           |Graph convolution   |0.872              |0.795              |
|           |Weave               |0.780              |0.773              |
|pcba       |Logistic regression |0.810              |0.742              |
|           |MT-NN classification|0.814              |0.760              |
|           |Robust MT-NN        |0.812              |0.756              |
|           |Graph convolution   |0.874              |0.817              |
|sider      |Logistic regression |0.926              |0.592              |
|           |Random forest       |0.999              |0.619              |
|           |XGBoost             |0.796              |0.560              |
|           |IRV                 |0.639              |0.599              |
|           |MT-NN classification|0.776              |0.557              |
|           |Robust MT-NN        |0.797              |0.560              |
|           |Graph convolution   |0.722              |0.583              |
|           |Weave               |0.600              |0.529              |
|tox21      |Logistic regression |0.900              |0.650              |
|           |Random forest       |0.999              |0.629              |
|           |XGBoost             |0.881              |0.703              |
|           |IRV                 |0.823              |0.708              |
|           |MT-NN classification|0.863              |0.703              |
|           |Robust MT-NN        |0.861              |0.710              |
|           |Graph convolution   |0.885              |0.732              |
|           |Weave               |0.866              |0.773              |
|toxcast    |Logistic regression |0.716              |0.492              |
|           |XGBoost             |0.741              |0.587              |
|           |MT-NN classification|0.828              |0.617              |
|           |Robust MT-NN        |0.830              |0.614              |
|           |Graph convolution   |0.832              |0.638              |
|           |Weave               |0.766              |0.637              |


* Regression

|Dataset         |Model               |Splitting   |Train score/R2|Valid score/R2|
|----------------|--------------------|------------|--------------|--------------|
|bace_r          |Random forest       |Random      |0.958         |0.646         |
|                |NN regression       |Random      |0.898         |0.680         |
|                |Graphconv regression|Random      |0.760         |0.676         |
|                |Weave regression    |Random      |0.523         |0.577         |
|                |Random forest       |Scaffold    |0.956         |0.201         |
|                |NN regression       |Scaffold    |0.897         |0.208         |
|                |Graphconv regression|Scaffold    |0.783         |0.068         |
|                |Weave regression    |Scaffold    |0.602         |0.018         |
|chembl          |MT-NN regression    |Index       |0.828         |0.565         |
|                |Graphconv regression|Index       |0.192         |0.293         |
|                |MT-NN regression    |Random      |0.829         |0.562         |
|                |Graphconv regression|Random      |0.198         |0.271         |
|                |MT-NN regression    |Scaffold    |0.843         |0.430         |
|                |Graphconv regression|Scaffold    |0.231         |0.294         |
|clearance       |Random forest       |Index       |0.953         |0.244         |
|                |NN regression       |Index       |0.884         |0.211         |
|                |Graphconv regression|Index       |0.696         |0.230         |
|                |Weave regression    |Index       |0.261         |0.107         |
|                |Random forest       |Random      |0.952         |0.547         |
|                |NN regression       |Random      |0.880         |0.273         |
|                |Graphconv regression|Random      |0.685         |0.302         |
|                |Weave regression    |Random      |0.229         |0.129         |
|                |Random forest       |Scaffold    |0.952         |0.266         |
|                |NN regression       |Scaffold    |0.871         |0.154         |
|                |Graphconv regression|Scaffold    |0.628         |0.277         |
|                |Weave regression    |Scaffold    |0.228         |0.226         |
|delaney         |Random forest       |Index       |0.953         |0.626         |
|                |XGBoost             |Index       |0.898         |0.664         |
|                |NN regression       |Index       |0.868         |0.578         |
|                |Graphconv regression|Index       |0.967         |0.790         |
|                |Weave regression    |Index       |0.965         |0.888         |
|                |Random forest       |Random      |0.951         |0.684         |
|                |XGBoost             |Random      |0.927         |0.727         |
|                |NN regression       |Random      |0.865         |0.574         |
|                |Graphconv regression|Random      |0.964         |0.782         |
|                |Weave regression    |Random      |0.954         |0.917         |
|                |Random forest       |Scaffold    |0.953         |0.284         |
|                |XGBoost             |Scaffold    |0.890         |0.316         |
|                |NN regression       |Scaffold    |0.866         |0.342         |
|                |Graphconv regression|Scaffold    |0.967         |0.606         |
|                |Weave regression    |Scaffold    |0.976         |0.797         |
|hopv            |Random forest       |Index       |0.943         |0.338         |
|                |MT-NN regression    |Index       |0.725         |0.293         |
|                |Graphconv regression|Index       |0.307         |0.284         |
|                |Weave regression    |Index       |0.046         |0.026         |
|                |Random forest       |Random      |0.943         |0.513         |
|                |MT-NN regression    |Random      |0.716         |0.289         |
|                |Graphconv regression|Random      |0.329         |0.239         |
|                |Weave regression    |Random      |0.080         |0.084         |
|                |Random forest       |Scaffold    |0.946         |0.470         |
|                |MT-NN regression    |Scaffold    |0.719         |0.429         |
|                |Graphconv regression|Scaffold    |0.286         |0.155         |
|                |Weave regression    |Scaffold    |0.097         |0.082         |
|kaggle          |MT-NN regression    |User-defined|0.748         |0.452         |
|lipo            |Random forest       |Index       |0.960         |0.483         |
|                |NN regression       |Index       |0.825         |0.513         |
|                |Graphconv regression|Index       |0.865         |0.704         |
|                |Weave regression    |Index       |0.507         |0.492         |
|                |Random forest       |Random      |0.958         |0.518         |
|                |NN regression       |Random      |0.818         |0.445         |
|                |Graphconv regression|Random      |0.867         |0.722         |
|                |Weave regression    |Random      |0.551         |0.528         |
|                |Random forest       |Scaffold    |0.958         |0.329         |
|                |NN regression       |Scaffold    |0.831         |0.302         |
|                |Graphconv regression|Scaffold    |0.882         |0.593         |
|                |Weave regression    |Scaffold    |0.566         |0.448         |
|nci             |XGBoost             |Index       |0.441         |0.066         |
|                |MT-NN regression    |Index       |0.690         |0.062         |
|                |Graphconv regression|Index       |0.123         |0.053         |
|                |XGBoost             |Random      |0.409         |0.106         |
|                |MT-NN regression    |Random      |0.698         |0.117         |
|                |Graphconv regression|Random      |0.117         |0.076         |
|                |XGBoost             |Scaffold    |0.445         |0.046         |
|                |MT-NN regression    |Scaffold    |0.692         |0.036         |
|                |Graphconv regression|Scaffold    |0.131         |0.036         |
|pdbbind(core)   |Random forest       |Random      |0.969         |0.445         |
|                |NN regression       |Random      |0.973         |0.494         |
|pdbbind(refined)|Random forest       |Random      |0.963         |0.511         |
|                |NN regression       |Random      |0.987         |0.503         |
|pdbbind(full)   |Random forest       |Random      |0.965         |0.493         |
|                |NN regression       |Random      |0.983         |0.528         |
|ppb             |Random forest       |Index       |0.951         |0.235         |
|                |NN regression       |Index       |0.902         |0.333         |
|                |Graphconv regression|Index       |0.673         |0.442         |
|                |Weave regression    |Index       |0.418         |0.301         |
|                |Random forest       |Random      |0.950         |0.220         |
|                |NN regression       |Random      |0.903         |0.244         |
|                |Graphconv regression|Random      |0.646         |0.429         |
|                |Weave regression    |Random      |0.408         |0.284         |
|                |Random forest       |Scaffold    |0.943         |0.176         |
|                |NN regression       |Scaffold    |0.902         |0.144         |
|                |Graphconv regression|Scaffold    |0.695         |0.391         |
|                |Weave regression    |Scaffold    |0.401         |0.373         |
|qm7             |NN regression       |Index       |0.997         |0.992         |
|                |DTNN                |Index       |0.997         |0.995         |
|                |NN regression       |Random      |0.998         |0.997         |
|                |DTNN                |Random      |0.999         |0.998         |
|                |NN regression       |Stratified  |0.998         |0.997         | 
|                |DTNN                |Stratified  |0.998         |0.998         | 
|qm7b            |MT-NN regression    |Index       |0.903         |0.789         |
|                |DTNN                |Index       |0.919         |0.863         |
|                |MT-NN regression    |Random      |0.893         |0.839         |
|                |DTNN                |Random      |0.924         |0.898         |
|                |MT-NN regression    |Stratified  |0.891         |0.859         | 
|                |DTNN                |Stratified  |0.913         |0.894         | 
|qm8             |MT-NN regression    |Index       |0.783         |0.656         |
|                |DTNN                |Index       |0.857         |0.691         |
|                |MT-NN regression    |Random      |0.747         |0.660         |
|                |DTNN                |Random      |0.842         |0.756         |
|                |MT-NN regression    |Stratified  |0.756         |0.681         |
|                |DTNN                |Stratified  |0.844         |0.758         | 
|qm9             |MT-NN regression    |Index       |0.733         |0.766         |
|                |DTNN                |Index       |0.918         |0.831         | 
|                |MT-NN regression    |Random      |0.852         |0.833         |
|                |DTNN                |Random      |0.942         |0.948         | 
|                |MT-NN regression    |Stratified  |0.764         |0.792         | 
|                |DTNN                |Stratified  |0.941         |0.867         | 
|sampl           |Random forest       |Index       |0.968         |0.736         |
|                |XGBoost             |Index       |0.884         |0.784         |
|                |NN regression       |Index       |0.917         |0.764         |
|                |Graphconv regression|Index       |0.982         |0.903         |
|                |Weave regression    |Index       |0.993         |0.948         | 
|                |Random forest       |Random      |0.967         |0.752         |
|                |XGBoost             |Random      |0.906         |0.745         |
|                |NN regression       |Random      |0.908         |0.711         |
|                |Graphconv regression|Random      |0.987         |0.868         |
|                |Weave regression    |Random      |0.992         |0.888         |
|                |Random forest       |Scaffold    |0.966         |0.477         |
|                |XGBoost             |Scaffold    |0.918         |0.439         |
|                |NN regression       |Scaffold    |0.891         |0.217         |
|                |Graphconv regression|Scaffold    |0.985         |0.666         |
|                |Weave regression    |Scaffold    |0.988         |0.876         |

|Dataset         |Model            |Splitting   |Train score/MAE(kcal/mol)|Valid score/MAE(kcal/mol)|
|----------------|-----------------|------------|-------------------------|-------------------------|
|qm7             |NN regression    |Index       |11.0                     |12.0                     |
|                |NN regression    |Random      |7.12                     |7.53                     |
|                |NN regression    |Stratified  |6.61                     |7.34                     |


* General features

Number of tasks and examples in the datasets

|Dataset         |N(tasks)   |N(samples) |
|----------------|-----------|-----------| 
|bace_c          |1          |1522       |
|bbbp            |1          |2053       |
|clintox         |2          |1491       |
|hiv             |1          |41913      |
|muv             |17         |93127      |
|pcba            |128        |439863     |
|sider           |27         |1427       |
|tox21           |12         |8014       |
|toxcast         |617        |8615       |
|bace_r          |1          |1522       |
|chembl(5thresh) |691        |23871      |
|clearance       |1          |837        |
|delaney         |1          |1128       |
|hopv            |8          |350        |
|kaggle          |15         |173065     |
|lipo            |1          |4200       |
|nci             |60         |19127      |
|pdbbind(core)   |1          |195        |
|pdbbind(refined)|1          |3706       |
|pdbbind(full)   |1          |11908      |
|ppb             |1          |1614       |
|qm7             |1          |7165       |
|qm7b            |14         |7211       |
|qm8             |16         |21786      |
|qm9             |15         |133885     |
|sampl           |1          |643        |


Time needed for benchmark test(~20h in total)

|Dataset         |Model               |Time(loading)/s |Time(running)/s|
|----------------|--------------------|----------------|---------------| 
|bace_c          |Logistic regression |10              |10             |
|                |NN classification   |10              |10             |
|                |Robust NN           |10              |10             |
|                |Random forest       |10              |80             |
|                |IRV                 |10              |10             |
|                |Graph convolution   |15              |70             |
|                |Weave               |15              |120            |
|bbbp            |Logistic regression |20              |10             |
|                |NN classification   |20              |20             |
|                |Robust NN           |20              |20             |
|                |Random forest       |20              |120            |
|                |IRV                 |20              |10             |
|                |Graph convolution   |20              |150            |
|                |Weave               |20              |100            |
|clintox         |Logistic regression |15              |10             |
|                |XGBoost             |15              |33             |
|                |MT-NN classification|15              |20             |
|                |Robust MT-NN        |15              |30             |
|                |Random forest       |15              |200            |
|                |IRV                 |15              |10             |
|                |Graph convolution   |20              |130            |
|                |Weave               |20              |90             |
|hiv             |Logistic regression |180             |40             |
|                |XGBoost             |180             |1000           |
|                |NN classification   |180             |350            |
|                |Robust NN           |180             |450            |
|                |Random forest       |180             |2800           |
|                |IRV                 |180             |200            |
|                |Graph convolution   |180             |1300           |
|                |Weave               |180             |2000           |
|muv             |Logistic regression |600             |450            |
|                |XGBoost             |600             |3500           |
|                |MT-NN classification|600             |400            |
|                |Robust MT-NN        |600             |550            |
|                |Graph convolution   |800             |1800           |
|                |Weave               |800             |4400           |
|pcba            |Logistic regression |1800            |10000          |
|                |XGBoost             |1800            |470000         |
|                |MT-NN classification|1800            |9000           |
|                |Robust MT-NN        |1800            |14000          |
|                |Graph convolution   |2200            |14000          |
|sider           |Logistic regression |15              |80             |
|                |XGBoost             |15              |660            |
|                |MT-NN classification|15              |75             |
|                |Robust MT-NN        |15              |150            |
|                |Random forest       |15              |2200           |
|                |IRV                 |15              |150            |
|                |Graph convolution   |20              |50             |
|                |Weave               |20              |200            |
|tox21           |Logistic regression |30              |60             |
|                |XGBoost             |30              |1500           |
|                |MT-NN classification|30              |60             |
|                |Robust MT-NN        |30              |90             |
|                |Random forest       |30              |6000           |
|                |IRV                 |30              |650            |
|                |Graph convolution   |30              |160            |
|                |Weave               |30              |300            |
|toxcast         |Logistic regression |80              |2600           |
|                |XGBoost             |80              |30000          |
|                |MT-NN classification|80              |2300           |
|                |Robust MT-NN        |80              |4000           |
|                |Graph convolution   |80              |900            |
|                |Weave               |80              |2000           |
|bace_r          |NN regression       |10              |30             |
|                |Random forest       |10              |50             |
|                |Graphconv regression|10              |110            |
|                |Weave regression    |10              |150            |
|chembl          |MT-NN regression    |200             |9000           |
|                |Graphconv regression|250             |1800           |
|clearance       |NN regression       |10              |20             |
|                |Random forest       |10              |10             |
|                |Graphconv regression|10              |60             |
|                |Weave regression    |10              |70             |
|delaney         |NN regression       |10              |40             |
|                |XGBoost             |10              |50             |
|                |Random forest       |10              |30             |
|                |graphconv regression|10              |40             |
|                |Weave regression    |10              |40             |
|hopv            |MT-NN regression    |10              |20             |
|                |Random forest       |10              |50             |
|                |Graphconv regression|10              |50             |
|                |Weave regression    |10              |60             |
|kaggle          |MT-NN regression    |2200            |3200           |
|lipo            |NN regression       |30              |60             |
|                |Random forest       |30              |60             |
|                |Graphconv regression|30              |240            |
|                |Weave regression    |30              |280            |
|nci             |MT-NN regression    |400             |1200           |
|                |XGBoost             |400             |28000          |
|                |graphconv regression|400             |2500           |
|pdbbind(core)   |NN regression       |0(featurized)   |30             |
|pdbbind(refined)|NN regression       |0(featurized)   |40             |
|pdbbind(full)   |NN regression       |0(featurized)   |60             |
|ppb             |NN regression       |20              |30             |
|                |Random forest       |20              |30             |
|                |Graphconv regression|20              |100            |
|                |Weave regression    |20              |120            |
|qm7             |MT-NN regression    |10              |400            |
|                |DTNN                |10              |600            |
|qm7b            |MT-NN regression    |10              |600            |
|                |DTNN                |10              |600            |
|qm8             |MT-NN regression    |60              |1000           |
|                |DTNN                |10              |2000           |
|qm9             |MT-NN regression    |220             |10000          |
|                |DTNN                |10              |14000          |
|sampl           |NN regression       |10              |30             |
|                |XGBoost             |10              |20             |
|                |Random forest       |10              |20             |
|                |graphconv regression|10              |40             |
|                |Weave regression    |10              |20             |



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
DeepChem is a package by the [Pande group](https://pande.stanford.edu/) at Stanford. DeepChem was originally created by [Bharath Ramsundar](http://rbharath.github.io/), and has grown through the contributions of a number of undergraduate, graduate, and postdoctoral researchers working with the Pande lab.

## Corporate Supporters
DeepChem is supported by a number of corporate partners who use DeepChem to solve interesting problems.

### Schrödinger
[![Schödinger](https://github.com/deepchem/deepchem/raw/master/docs/_static/schrodinger_logo.png)](https://www.schrodinger.com/)

> DeepChem has transformed how we think about building QSAR and QSPR models when very large data sets are available; and we are actively using DeepChem to investigate how to best combine the power of deep learning with next generation physics-based scoring methods.

### DeepCrystal
<img src="https://raw.githubusercontent.com/deepchem/deepchem/master/docs/_static/deep_crystal_logo.png" alt="DeepCrystal Logo" height=150px/>

> DeepCrystal was an early adopter of DeepChem, which we now rely on to abstract away some of the hardest pieces of deep learning in drug discovery. By open sourcing these efficient implementations of chemically / biologically aware deep-learning systems, DeepChem puts the latest research into the hands of the scientists that need it, materially pushing forward the field of in-silico drug discovery in the process.


## Version
1.1.0

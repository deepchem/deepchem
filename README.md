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
pip install tensorflow-gpu==1.3.0                      # If you want GPU support
python setup.py install                                 # Manual install
nosetests -v deepchem --nologcapture                    # Run tests
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
* Classification

Index splitting

|Dataset    |Model               |Train score/ROC-AUC|Valid score/ROC-AUC|
|-----------|--------------------|-------------------|-------------------|
|clintox    |Logistic regression |0.969              |0.683              |
|           |Random forest       |0.995              |0.763              |
|           |XGBoost             |0.879              |0.890              |
|           |IRV                 |0.762              |0.811              |
|           |MT-NN classification|0.929              |0.832              |
|           |Robust MT-NN        |0.948              |0.840              |
|           |Graph convolution   |0.961              |0.812              |
|           |DAG                 |0.997              |0.660              |
|           |Weave               |0.937              |0.887              |
|hiv        |Logistic regression |0.861              |0.731              |
|           |Random forest       |0.999              |0.720              |
|           |XGBoost             |0.917              |0.745              |
|           |IRV                 |0.841              |0.724              |
|           |NN classification   |0.712              |0.676              |
|           |Robust NN           |0.740              |0.699              |
|           |Graph convolution   |0.888              |0.771              |
|           |Weave               |0.880              |0.758              |
|muv        |Logistic regression |0.957              |0.754              |
|           |XGBoost             |0.895              |0.714              |
|           |MT-NN classification|0.900              |0.746              |
|           |Robust MT-NN        |0.937              |0.765              |
|           |Graph convolution   |0.890              |0.804              |
|           |Weave               |0.749              |0.764              |
|pcba       |Logistic regression |0.807              |0.773              |
|           |XGBoost             |0.931              |0.847              |
|           |MT-NN classification|0.819              |0.792              |
|           |Robust MT-NN        |0.812              |0.782              |
|           |Graph convolution   |0.886              |0.851              |
|sider      |Logistic regression |0.932              |0.622              |
|           |Random forest       |1.000              |0.669              |
|           |XGBoost             |0.829              |0.639              |
|           |IRV                 |0.649              |0.643              |
|           |MT-NN classification|0.781              |0.630              |
|           |Robust MT-NN        |0.805              |0.634              |
|           |Graph convolution   |0.744              |0.593              |
|           |DAG                 |0.908              |0.558              |
|           |Weave               |0.622              |0.599              |
|tox21      |Logistic regression |0.902              |0.705              |
|           |Random forest       |0.999              |0.736              |
|           |XGBoost             |0.891              |0.753              |
|           |IRV                 |0.811              |0.767              |
|           |MT-NN classification|0.854              |0.768              |
|           |Robust MT-NN        |0.857              |0.766              |
|           |Graph convolution   |0.903              |0.814              |
|           |DAG                 |0.871              |0.733              |
|           |Weave               |0.844              |0.797              |
|toxcast    |Logistic regression |0.724              |0.577              |
|           |XGBoost             |0.738              |0.621              |
|           |IRV                 |0.662              |0.643              |
|           |MT-NN classification|0.830              |0.684              |
|           |Robust MT-NN        |0.825              |0.681              |
|           |Graph convolution   |0.849              |0.726              |
|           |Weave               |0.796              |0.725              |

Random splitting

|Dataset    |Model               |Train score/ROC-AUC|Valid score/ROC-AUC|
|-----------|--------------------|-------------------|-------------------|
|bace_c     |Logistic regression |0.952              |0.860              |
|           |Random forest       |1.000              |0.882              |
|           |IRV                 |0.876              |0.871              |
|           |NN classification   |0.868              |0.838              |
|           |Robust NN           |0.892              |0.853              |
|           |Graph convolution   |0.849              |0.793              |
|           |DAG                 |0.873              |0.810              |
|           |Weave               |0.828              |0.847              |
|bbbp       |Logistic regression |0.978              |0.905              |
|           |Random forest       |1.000              |0.908              |
|           |IRV                 |0.912              |0.889              |
|           |NN classification   |0.857              |0.822              |
|           |Robust NN           |0.886              |0.857              |
|           |Graph convolution   |0.966              |0.870              |
|           |DAG                 |0.986              |0.888              |
|           |Weave               |0.935              |0.898              |
|clintox    |Logistic regression |0.968              |0.734              |
|           |Random forest       |0.996              |0.730              |
|           |XGBoost             |0.886              |0.731              |
|           |IRV                 |0.793              |0.751              |
|           |MT-NN classification|0.946              |0.793              |
|           |Robust MT-NN        |0.958              |0.818              |
|           |Graph convolution   |0.965              |0.908              |
|           |DAG                 |0.998              |0.529              |
|           |Weave               |0.927              |0.867              |
|hiv        |Logistic regression |0.855              |0.816              |
|           |Random forest       |0.999              |0.850              |
|           |XGBoost             |0.933              |0.841              |
|           |IRV                 |0.831              |0.836              |
|           |NN classification   |0.699              |0.695              |
|           |Robust NN           |0.726              |0.726              |
|           |Graph convolution   |0.876              |0.824              |
|           |Weave               |0.872              |0.819              |
|muv        |Logistic regression |0.954              |0.722              |
|           |XGBoost             |0.874              |0.696              |
|           |IRV                 |0.690              |0.630              |
|           |MT-NN classification|0.906              |0.737              |
|           |Robust MT-NN        |0.940              |0.732              |
|           |Graph convolution   |0.889              |0.734              |
|           |Weave               |0.757              |0.714              |
|pcba       |Logistic regression |0.808        	     |0.775              |
|           |MT-NN classification|0.811        	     |0.787              |
|           |Robust MT-NN        |0.809              |0.776              |
|           |Graph convolution   |0.888       	     |0.850              |
|sider      |Logistic regression |0.931        	     |0.639              |
|           |Random forest       |1.000              |0.682              |
|           |XGBoost             |0.824              |0.635              |
|           |IRV                 |0.636              |0.634              |
|           |MT-NN classification|0.782        	     |0.662              |
|           |Robust MT-NN        |0.807              |0.661              |
|           |Graph convolution   |0.732        	     |0.666              |
|           |DAG                 |0.919              |0.555              |
|           |Weave               |0.597              |0.610              |
|tox21      |Logistic regression |0.900              |0.735              |
|           |Random forest       |0.999              |0.763              |
|           |XGBoost             |0.874              |0.773              |
|           |IRV                 |0.807              |0.770              |
|           |MT-NN classification|0.849              |0.754              |
|           |Robust MT-NN        |0.854              |0.755              |
|           |Graph convolution   |0.901              |0.832              |
|           |DAG                 |0.888              |0.766              |
|           |Weave               |0.844              |0.812              |
|toxcast    |Logistic regression |0.719        	     |0.538              |
|           |XGBoost             |0.738              |0.633              |
|           |IRV                 |0.659              |0.662              |
|           |MT-NN classification|0.836        	     |0.676              |
|           |Robust MT-NN        |0.828              |0.680              |
|           |Graph convolution   |0.843        	     |0.732              |
|           |Weave               |0.785              |0.718              |

Scaffold splitting

|Dataset    |Model               |Train score/ROC-AUC|Valid score/ROC-AUC|
|-----------|--------------------|-------------------|-------------------|
|bace_c     |Logistic regression |0.957              |0.726              |
|           |Random forest       |0.999              |0.728              |
|           |IRV                 |0.899              |0.700              |
|           |NN classification   |0.884              |0.710              |
|           |Robust NN           |0.906              |0.738              |
|           |Graph convolution   |0.921              |0.665              |
|           |DAG                 |0.839              |0.591              |
|           |Weave               |0.736              |0.593              |
|bbbp       |Logistic regression |0.980              |0.957              |
|           |Random forest       |1.000              |0.955              |
|           |IRV                 |0.914              |0.962              |
|           |NN classification   |0.884              |0.955              |
|           |Robust NN           |0.905              |0.959              |
|           |Graph convolution   |0.972              |0.949              |
|           |DAG                 |0.940              |0.855              |
|           |Weave               |0.953              |0.969              |
|clintox    |Logistic regression |0.962              |0.687              |
|           |Random forest       |0.994              |0.664              |
|           |XGBoost             |0.873              |0.850              |
|           |IRV                 |0.793              |0.715              |
|           |MT-NN classification|0.923              |0.825              |
|           |Robust MT-NN        |0.949              |0.821              |
|           |Graph convolution   |0.973              |0.847              |
|           |DAG                 |0.991              |0.451              |
|           |Weave               |0.936              |0.930              |
|hiv        |Logistic regression |0.858              |0.793              |
|           |Random forest       |0.946              |0.562              |
|           |XGBoost             |0.927              |0.830              |
|           |IRV                 |0.847              |0.811              |
|           |NN classification   |0.719              |0.718              |
|           |Robust NN           |0.740              |0.730              |
|           |Graph convolution   |0.882              |0.797              |
|           |Weave               |0.880              |0.793              |
|muv        |Logistic regression |0.950              |0.756              |
|           |XGBoost             |0.875              |0.705              |
|           |IRV                 |0.666              |0.708              |
|           |MT-NN classification|0.908              |0.785              |
|           |Robust MT-NN        |0.934              |0.792              |
|           |Graph convolution   |0.899              |0.787              |
|           |Weave               |0.762              |0.764              |
|pcba       |Logistic regression |0.810              |0.748              |
|           |MT-NN classification|0.823              |0.773              |
|           |Robust MT-NN        |0.818              |0.758              |
|           |Graph convolution   |0.894              |0.826              |
|sider      |Logistic regression |0.926              |0.594              |
|           |Random forest       |1.000              |0.611              |
|           |XGBoost             |0.796              |0.560              |
|           |IRV                 |0.638              |0.598              |
|           |MT-NN classification|0.771              |0.555              |
|           |Robust MT-NN        |0.795              |0.567              |
|           |Graph convolution   |0.751              |0.546              |
|           |DAG                 |0.902              |0.541              |
|           |Weave               |0.640              |0.509              |
|tox21      |Logistic regression |0.901              |0.676              |
|           |Random forest       |0.999              |0.665              |
|           |XGBoost             |0.881              |0.703              |
|           |IRV                 |0.823              |0.708              |
|           |MT-NN classification|0.863              |0.725              |
|           |Robust MT-NN        |0.861              |0.724              |
|           |Graph convolution   |0.913              |0.764              |
|           |DAG                 |0.888              |0.658              |
|           |Weave               |0.864              |0.763              |
|toxcast    |Logistic regression |0.717              |0.511              |
|           |XGBoost             |0.741              |0.587              |
|           |IRV                 |0.677              |0.612              |
|           |MT-NN classification|0.835              |0.612              |
|           |Robust MT-NN        |0.832              |0.609              |
|           |Graph convolution   |0.859              |0.646              |
|           |Weave               |0.802              |0.657              |


* Regression

|Dataset         |Model               |Splitting   |Train score/R2|Valid score/R2|
|----------------|--------------------|------------|--------------|--------------|
|bace_r          |Random forest       |Random      |0.958         |0.680         |
|                |NN regression       |Random      |0.895         |0.732         |
|                |Graphconv regression|Random      |0.328         |0.276         |
|                |DAG regression      |Random      |0.370         |0.271         |
|                |Weave regression    |Random      |0.555         |0.578         |
|                |Random forest       |Scaffold    |0.956         |0.203         |
|                |NN regression       |Scaffold    |0.894         |0.203         |
|                |Graphconv regression|Scaffold    |0.321         |0.032         |
|                |DAG regression      |Scaffold    |0.304         |0.000         |
|                |Weave regression    |Scaffold    |0.594         |0.044         |
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
|delaney         |Random forest       |Index       |0.954         |0.625         |
|                |XGBoost             |Index       |0.898         |0.664         |
|                |NN regression       |Index       |0.869         |0.585         |
|                |Graphconv regression|Index       |0.969         |0.813         |
|                |DAG regression      |Index       |0.976         |0.850         |
|                |Weave regression    |Index       |0.963         |0.872         |
|                |Random forest       |Random      |0.955         |0.561         |
|                |XGBoost             |Random      |0.927         |0.727         |
|                |NN regression       |Random      |0.875         |0.495         |
|                |Graphconv regression|Random      |0.976         |0.787         |
|                |DAG regression      |Random      |0.968         |0.899         |
|                |Weave regression    |Random      |0.955         |0.907         |
|                |Random forest       |Scaffold    |0.953         |0.281         |
|                |XGBoost             |Scaffold    |0.890         |0.316         |
|                |NN regression       |Scaffold    |0.872         |0.308         |
|                |Graphconv regression|Scaffold    |0.980         |0.564         |
|                |DAG regression      |Scaffold    |0.968         |0.676         |
|                |Weave regression    |Scaffold    |0.971         |0.756         |
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
|lipo            |Random forest       |Index       |0.960         |0.485         |
|                |NN regression       |Index       |0.829         |0.508         |
|                |Graphconv regression|Index       |0.867         |0.702         |
|                |DAG regression      |Index       |0.957         |0.483         |
|                |Weave regression    |Index       |0.726         |0.607         |
|                |Random forest       |Random      |0.960         |0.514         |
|                |NN regression       |Random      |0.833         |0.476         |
|                |Graphconv regression|Random      |0.867         |0.631         |
|                |DAG regression      |Random      |0.967         |0.412         |
|                |Weave regression    |Random      |0.747         |0.598         |
|                |Random forest       |Scaffold    |0.959         |0.330         |
|                |NN regression       |Scaffold    |0.830         |0.308         |
|                |Graphconv regression|Scaffold    |0.875         |0.608         |
|                |DAG regression      |Scaffold    |0.937         |0.368         |
|                |Weave regression    |Scaffold    |0.761         |0.575         |
|nci             |XGBoost             |Index       |0.441         |0.066         |
|                |MT-NN regression    |Index       |0.690         |0.062         |
|                |Graphconv regression|Index       |0.123         |0.053         |
|                |XGBoost             |Random      |0.409         |0.106         |
|                |MT-NN regression    |Random      |0.698         |0.117         |
|                |Graphconv regression|Random      |0.117         |0.076         |
|                |XGBoost             |Scaffold    |0.445         |0.046         |
|                |MT-NN regression    |Scaffold    |0.692         |0.036         |
|                |Graphconv regression|Scaffold    |0.131         |0.036         |
|pdbbind(core)   |Random forest       |Random      |0.921         |0.382         |
|                |NN regression       |Random      |0.764         |0.591         |
|                |Graphconv regression|Random      |0.774         |0.230         |
|                |Random forest(grid) |Random      |0.970         |0.401         |
|                |NN regression(grid) |Random      |0.986         |0.180         |
|pdbbind(refined)|Random forest       |Random      |0.901         |0.562         |
|                |NN regression       |Random      |0.766         |0.442         |
|                |Graphconv regression|Random      |0.694         |0.508         |
|                |Random forest(grid) |Random      |0.963         |0.530         |
|                |NN regression(grid) |Random      |0.982         |0.484         |
|pdbbind(full)   |Random forest       |Random      |0.879         |0.475         |
|                |NN regression       |Random      |0.311         |0.307         |
|                |Graphconv regression|Random      |0.183         |0.186         |
|                |Random forest(grid) |Random      |0.966         |0.524         |
|                |NN regression(grid) |Random      |0.961         |0.492         |
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
|qm7             |Random forest       |Index       |0.942         |0.029         |
|                |NN regression       |Index       |0.782         |0.038         |
|                |Graphconv regression|Index       |0.982         |0.036         |
|                |NN regression(CM)   |Index       |0.997         |0.989         |
|                |DTNN                |Index       |0.998         |0.997         |
|                |Random forest       |Random      |0.935         |0.429         |
|                |NN regression       |Random      |0.643         |0.554         |
|                |Graphconv regression|Random      |0.892         |0.740         |
|                |NN regression(CM)   |Random      |0.997         |0.997         |
|                |DTNN                |Random      |0.998         |0.995         |
|                |Random forest       |Stratified  |0.934         |0.430         |
|                |NN regression       |Stratified  |0.630         |0.563         |
|                |Graphconv regression|Stratified  |0.894         |0.725         |
|                |NN regression(CM)   |Stratified  |0.998         |0.997         | 
|                |DTNN                |Stratified  |0.999         |0.998         | 
|qm7b            |MT-NN regression(CM)|Index       |0.900         |0.783         |
|                |DTNN                |Index       |0.926         |0.869         |
|                |MT-NN regression(CM)|Random      |0.891         |0.849         |
|                |DTNN                |Random      |0.925         |0.902         |
|                |MT-NN regression(CM)|Stratified  |0.892         |0.862         | 
|                |DTNN                |Stratified  |0.922         |0.905         | 
|qm8             |Random forest       |Index       |0.972         |0.616         |
|                |MT-NN regression    |Index       |0.939         |0.604         |
|                |Graphconv regression|Index       |0.866         |0.704         |
|                |MT-NN regression(CM)|Index       |0.770         |0.625         |
|                |DTNN                |Index       |0.856         |0.696         |
|                |Random forest       |Random      |0.971         |0.706         |
|                |MT-NN regression    |Random      |0.934         |0.717         |
|                |Graphconv regression|Random      |0.848         |0.780         |
|                |MT-NN regression(CM)|Random      |0.753         |0.699         |
|                |DTNN                |Random      |0.842         |0.754         |
|                |Random forest       |Stratified  |0.971         |0.690         |
|                |MT-NN regression    |Stratified  |0.934         |0.712         |
|                |Graphconv regression|Stratified  |0.846         |0.767         |
|                |MT-NN regression(CM)|Stratified  |0.761         |0.696         |
|                |DTNN                |Stratified  |0.846         |0.745         |
|qm9             |MT-NN regression    |Index       |0.839         |0.708         |
|                |Graphconv regression|Index       |0.754         |0.768         |
|                |MT-NN regression(CM)|Index       |0.803         |0.800         |
|                |DTNN                |Index       |0.911         |0.867         | 
|                |MT-NN regression    |Random      |0.849         |0.753         |
|                |Graphconv regression|Random      |0.700         |0.696         |
|                |MT-NN regression(CM)|Random      |0.822         |0.823         |
|                |DTNN                |Random      |0.913         |0.867         | 
|                |MT-NN regression    |Stratified  |0.839         |0.687         |
|                |Graphconv regression|Stratified  |0.724         |0.696         |
|                |MT-NN regression(CM)|Stratified  |0.791         |0.827         | 
|                |DTNN                |Stratified  |0.911         |0.874         | 
|sampl           |Random forest       |Index       |0.967         |0.737         |
|                |XGBoost             |Index       |0.884         |0.784         |
|                |NN regression       |Index       |0.923         |0.758         |
|                |Graphconv regression|Index       |0.970         |0.897         |
|                |DAG regression      |Index       |0.970         |0.871         |
|                |Weave regression    |Index       |0.992         |0.915         | 
|                |Random forest       |Random      |0.966         |0.729         |
|                |XGBoost             |Random      |0.906         |0.745         |
|                |NN regression       |Random      |0.931         |0.689         |
|                |Graphconv regression|Random      |0.964         |0.848         |
|                |DAG regression      |Random      |0.973         |0.861         |
|                |Weave regression    |Random      |0.992         |0.885         |
|                |Random forest       |Scaffold    |0.967         |0.465         |
|                |XGBoost             |Scaffold    |0.918         |0.439         |
|                |NN regression       |Scaffold    |0.901         |0.238         |
|                |Graphconv regression|Scaffold    |0.963         |0.822         |
|                |DAG regression      |Scaffold    |0.961         |0.846         |
|                |Weave regression    |Scaffold    |0.992         |0.837         |

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

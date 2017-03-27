# DeepChem
[![Build Status](https://travis-ci.org/deepchem/deepchem.svg?branch=master)](https://travis-ci.org/deepchem/deepchem)
[![Coverage Status](https://coveralls.io/repos/github/deepchem/deepchem/badge.svg?branch=master)](https://coveralls.io/github/deepchem/deepchem?branch=master)

DeepChem aims to provide a high quality open-source toolchain that
democratizes the use of deep-learning in drug discovery, materials science, and quantum
chemistry. DeepChem is a package developed by the [Pande group](https://pande.stanford.edu/) at
Stanford and originally created by [Bharath Ramsundar](http://rbharath.github.io/).

### Table of contents:

* [Requirements](#requirements)
* [Installation from Source](#installation)
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
bash scripts/install_deepchem_conda.sh deepchem
pip install tensorflow-gpu==1.0.1                      # If you want GPU support
git clone https://github.com/deepchem/deepchem.git      # Clone deepchem source code from GitHub
cd deepchem
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
    pip install tensorflow-gpu==1.0.1
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
2. Question: The test suite is core-dumping for me. What's up?
   ```
   [rbharath]$ nosetests -v deepchem --nologcapture
   Illegal instruction (core dumped)
   ```
   
   Answer: This is often due to `openbabel` issues on older linux systems. Open `ipython` and run the following
   ```
   In [1]: import openbabel as ob
   ```
   If you see a core-dump, then it's a sign there's an issue with your `openbabel` install. Try reinstalling `openbabel` from source for your machine.
   
   
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
|tox21      |logistic regression |0.903              |0.705              |
|           |Random Forest       |0.999              |0.733              |
|           |IRV                 |0.811              |0.767              |
|           |Multitask network   |0.856              |0.763              |
|           |robust MT-NN        |0.857              |0.767              |
|           |graph convolution   |0.872              |0.798              |
|muv        |logistic regression |0.963              |0.766              |
|           |Multitask network   |0.904              |0.764              |
|           |robust MT-NN        |0.934              |0.781              |
|           |graph convolution   |0.840              |0.823              |
|pcba       |logistic regression |0.809              |0.776              |
|           |Multitask network   |0.826              |0.802              |
|           |robust MT-NN        |0.809              |0.783              |
|           |graph convolution   |0.876              |0.852              |
|sider      |logistic regression |0.933              |0.620              |
|           |Random Forest       |0.999              |0.670              |
|           |IRV                 |0.649              |0.642              |
|           |Multitask network   |0.775              |0.634              |
|           |robust MT-NN        |0.803              |0.632              |
|           |graph convolution   |0.708              |0.594              |
|toxcast    |logistic regression |0.721              |0.575              |
|           |Multitask network   |0.830              |0.678              |
|           |robust MT-NN        |0.825              |0.680              |
|           |graph convolution   |0.821              |0.720              |
|clintox    |logistic regression |0.967              |0.676              |
|           |Random Forest       |0.995              |0.776              |
|           |IRV                 |0.763              |0.814              |
|           |Multitask network   |0.934              |0.830              |
|           |robust MT-NN        |0.949              |0.827              |
|           |graph convolution   |0.946              |0.860              |
|hiv        |logistic regression |0.864              |0.739              |
|           |Random Forest       |0.999              |0.720              |
|           |IRV                 |0.841              |0.724              |
|           |Multitask network   |0.761              |0.652              |
|           |robust MT-NN        |0.780              |0.708              |
|           |graph convolution   |0.876              |0.779              |

Random splitting

|Dataset    |Model               |Train score/ROC-AUC|Valid score/ROC-AUC|
|-----------|--------------------|-------------------|-------------------|
|tox21      |logistic regression |0.902              |0.715              |
|           |Random Forest       |0.999              |0.764              |
|           |IRV                 |0.808              |0.767              |
|           |Multitask network   |0.844              |0.795              |
|           |robust MT-NN        |0.855              |0.773              |
|           |graph convolution   |0.865              |0.827              |
|muv        |logistic regression |0.957              |0.719              |
|           |Multitask network   |0.902              |0.734              |
|           |robust MT-NN        |0.933              |0.732              |
|           |graph convolution   |0.860              |0.730              |
|pcba       |logistic regression |0.808        	     |0.776              |
|           |Multitask network   |0.811        	     |0.778              |
|           |robust MT-NN        |0.811              |0.771              |
|           |graph convolution   |0.872       	     |0.844              |
|sider      |logistic regression |0.929        	     |0.656              |
|           |Random Forest       |0.999              |0.665              |
|           |IRV                 |0.648              |0.596              |
|           |Multitask network   |0.777        	     |0.655              |
|           |robust MT-NN        |0.804              |0.630              |
|           |graph convolution   |0.705        	     |0.618              |
|toxcast    |logistic regression |0.725        	     |0.586              |
|           |Multitask network   |0.836        	     |0.684              |
|           |robust MT-NN        |0.822              |0.681              |
|           |graph convolution   |0.820        	     |0.717              |
|clintox    |logistic regression |0.972              |0.725              |
|           |Random Forest       |0.997              |0.670              |
|           |IRV                 |0.809              |0.846              |
|           |Multitask network   |0.951              |0.834              |
|           |robust MT-NN        |0.959              |0.830              |
|           |graph convolution   |0.975              |0.876              |
|hiv        |logistic regression |0.860              |0.806              |
|           |Random Forest       |0.999              |0.850              |
|           |IRV                 |0.839              |0.809              |
|           |Multitask network   |0.742              |0.715              |
|           |robust MT-NN        |0.753              |0.727              |
|           |graph convolution   |0.847              |0.803              |

Scaffold splitting

|Dataset    |Model               |Train score/ROC-AUC|Valid score/ROC-AUC|
|-----------|--------------------|-------------------|-------------------|
|tox21      |logistic regression |0.900              |0.650              |
|           |Random Forest       |0.999              |0.629              |
|           |IRV                 |0.823              |0.708              |
|           |Multitask network   |0.863              |0.703              |
|           |robust MT-NN        |0.861              |0.710              |
|           |graph convolution   |0.885              |0.732              |
|muv        |logistic regression |0.947              |0.767              |
|           |Multitask network   |0.899              |0.762              |
|           |robust MT-NN        |0.944              |0.726              |
|           |graph convolution   |0.872              |0.795              |
|pcba       |logistic regression |0.810              |0.742              |
|           |Multitask network   |0.814              |0.760              |
|           |robust MT-NN        |0.812              |0.756              |
|           |graph convolution   |0.874              |0.817              |
|sider      |logistic regression |0.926              |0.592              |
|           |Random Forest       |0.999              |0.619              |
|           |IRV                 |0.639              |0.599              |
|           |Multitask network   |0.776              |0.557              |
|           |robust MT-NN        |0.797              |0.560              |
|           |graph convolution   |0.722              |0.583              |
|toxcast    |logistic regression |0.716              |0.492              |
|           |Multitask network   |0.828              |0.617              |
|           |robust MT-NN        |0.830              |0.614              |
|           |graph convolution   |0.832              |0.638              |
|clintox    |logistic regression |0.960              |0.803              |
|           |Random Forest       |0.993              |0.735              |
|           |IRV                 |0.793              |0.718              |
|           |Multitask network   |0.947              |0.862              |
|           |robust MT-NN        |0.953              |0.890              |
|           |graph convolution   |0.957              |0.823              |
|hiv        |logistic regression |0.858              |0.798              |
|           |Random Forest       |0.946              |0.562              |
|           |IRV                 |0.847              |0.811              |
|           |Multitask network   |0.775              |0.765              |
|           |robust MT-NN        |0.785              |0.748              |
|           |graph convolution   |0.867              |0.769              |

* Regression

|Dataset         |Model               |Splitting   |Train score/R2|Valid score/R2|
|----------------|--------------------|------------|--------------|--------------|
|delaney         |Random Forest       |Index       |0.953         |0.626         |
|                |NN regression       |Index       |0.868         |0.578         |
|                |graphconv regression|Index       |0.967         |0.790         |
|                |Random Forest       |Random      |0.951         |0.684         |
|                |NN regression       |Random      |0.865         |0.574         |
|                |graphconv regression|Random      |0.964         |0.782         |
|                |Random Forest       |Scaffold    |0.953         |0.284         |
|                |NN regression       |Scaffold    |0.866         |0.342         |
|                |graphconv regression|Scaffold    |0.967         |0.606         |
|sampl           |Random Forest       |Index       |0.968         |0.736         |
|                |NN regression       |Index       |0.917         |0.764         |
|                |graphconv regression|Index       |0.982         |0.864         |
|                |Random Forest       |Random      |0.967         |0.752         |
|                |NN regression       |Random      |0.908         |0.830         |
|                |graphconv regression|Random      |0.987         |0.868         |
|                |Random Forest       |Scaffold    |0.966         |0.473         |
|                |NN regression       |Scaffold    |0.891         |0.217         |
|                |graphconv regression|Scaffold    |0.985         |0.666         |
|nci             |NN regression       |Index       |0.171         |0.062         |
|                |graphconv regression|Index       |0.123         |0.048         |
|                |NN regression       |Random      |0.168         |0.085         |
|                |graphconv regression|Random      |0.117         |0.076         |
|                |NN regression       |Scaffold    |0.180         |0.052         |
|                |graphconv regression|Scaffold    |0.131         |0.046         |
|pdbbind(core)   |Random Forest       |Random      |0.969         |0.445         |
|                |NN regression       |Random      |0.973         |0.494         |
|pdbbind(refined)|Random Forest       |Random      |0.963         |0.511         |
|                |NN regression       |Random      |0.987         |0.503         |
|pdbbind(full)   |Random Forest       |Random      |0.965         |0.493         |
|                |NN regression       |Random      |0.983         |0.528         |
|chembl          |MT-NN regression    |Index       |0.443         |0.427         |
|                |MT-NN regression    |Random      |0.464         |0.434         |
|                |MT-NN regression    |Scaffold    |0.484         |0.361         |
|qm7             |NN regression       |Index       |0.997         |0.986         |
|                |NN regression       |Random      |0.999         |0.999         |
|                |NN regression       |Stratified  |0.999         |0.999         | 
|qm7b            |MT-NN regression    |Index       |0.931         |0.803         |
|                |MT-NN regression    |Random      |0.923         |0.884         |
|                |MT-NN regression    |Stratified  |0.934         |0.884         | 
|qm9             |MT-NN regression    |Index       |0.733         |0.791         |
|                |MT-NN regression    |Random      |0.811         |0.823         |
|                |MT-NN regression    |Stratified  |0.843         |0.818         | 
|kaggle          |MT-NN regression    |User-defined|0.748         |0.452         |

|Dataset         |Model            |Splitting   |Train score/MAE(kcal/mol)|Valid score/MAE(kcal/mol)|
|----------------|-----------------|------------|-------------------------|-------------------------|
|qm7             |NN regression    |Index       |11.0                     |12.0                     |
|                |NN regression    |Random      |7.12                     |7.53                     |
|                |NN regression    |Stratified  |6.61                     |7.34                     |


* General features

Number of tasks and examples in the datasets

|Dataset         |N(tasks)   |N(samples) |
|----------------|-----------|-----------| 
|tox21           |12         |8014       |
|muv             |17         |93127      |
|pcba            |128        |439863     |
|sider           |27         |1427       |
|toxcast         |617        |8615       |
|clintox         |2          |1491       |
|hiv             |1          |41913      |
|delaney         |1          |1128       |
|sampl           |1          |643        |
|kaggle          |15         |173065     |
|nci             |60         |19127      |
|pdbbind(core)   |1          |195        |
|pdbbind(refined)|1          |3706       |
|pdbbind(full)   |1          |11908      |
|chembl(5thresh) |691        |23871      |
|qm7             |1          |7165       |
|qm7b            |14         |7211       |
|qm9             |15         |133885     |


Time needed for benchmark test(~20h in total)

|Dataset         |Model               |Time(loading)/s |Time(running)/s|
|----------------|--------------------|----------------|---------------| 
|tox21           |logistic regression |30              |60             |
|                |Multitask network   |30              |60             |
|                |robust MT-NN        |30              |90             |
|                |random forest       |30              |6000           |
|                |IRV                 |30              |650            |
|                |graph convolution   |40              |160            |
|muv             |logistic regression |600             |450            |
|                |Multitask network   |600             |400            |
|                |robust MT-NN        |600             |550            |
|                |graph convolution   |800             |1800           |
|pcba            |logistic regression |1800            |10000          |
|                |Multitask network   |1800            |9000           |
|                |robust MT-NN        |1800            |14000          |
|                |graph convolution   |2200            |14000          |
|sider           |logistic regression |15              |80             |
|                |Multitask network   |15              |75             |
|                |robust MT-NN        |15              |150            |
|                |random forest       |15              |2200           |
|                |IRV                 |15              |150            |
|                |graph convolution   |20              |50             |
|toxcast         |logistic regression |80              |2600           |
|                |Multitask network   |80              |2300           |
|                |robust MT-NN        |80              |4000           |
|                |graph convolution   |80              |900            |
|clintox         |logistic regression |15              |10             |
|                |Multitask network   |15              |20             |
|                |robust MT-NN        |15              |30             |
|                |random forest       |15              |200            |
|                |IRV                 |15              |10             |
|                |graph convolution   |20              |130            |
|hiv             |logistic regression |180             |40             |
|                |Multitask network   |180             |350            |
|                |robust MT-NN        |180             |450            |
|                |random forest       |180             |2800           |
|                |IRV                 |180             |200            |
|                |graph convolution   |180             |1300           |
|delaney         |MT-NN regression    |10              |40             |
|                |graphconv regression|10              |40             |
|                |random forest       |10              |30             |
|sampl           |MT-NN regression    |10              |30             |
|                |graphconv regression|10              |40             |
|                |random forest       |10              |20             |
|nci             |MT-NN regression    |400             |1200           |
|                |graphconv regression|400             |2500           |
|pdbbind(core)   |MT-NN regression    |0(featurized)   |30             |
|pdbbind(refined)|MT-NN regression    |0(featurized)   |40             |
|pdbbind(full)   |MT-NN regression    |0(featurized)   |60             |
|chembl          |MT-NN regression    |200             |9000           |
|qm7             |MT-NN regression    |10              |400            |
|qm7b            |MT-NN regression    |10              |600            |
|qm9             |MT-NN regression    |220             |10000          |
|kaggle          |MT-NN regression    |2200            |3200           |



### Gitter
Join us on gitter at [https://gitter.im/deepchem/Lobby](https://gitter.im/deepchem/Lobby). Probably the easiest place to ask simple questions or float requests for new features.

## DeepChem Publications
1. [Computational Modeling of β-secretase 1 (BACE-1) Inhibitors using
Ligand Based
Approaches](http://pubs.acs.org/doi/abs/10.1021/acs.jcim.6b00290)
2. [Low Data Drug Discovery with One-shot Learning](https://arxiv.org/abs/1611.03199)
3. [MoleculeNet: A Benchmark for Molecular Machine Learning](https://arxiv.org/abs/1703.00564)

## About Us
DeepChem is a package by the [Pande group](https://pande.stanford.edu/) at Stanford. DeepChem was originally created by [Bharath Ramsundar](http://rbharath.github.io/), and has grown through the contributions of a number of undergraduate, graduate, and postdoctoral researchers working with the Pande lab.


## Version
1.0.1

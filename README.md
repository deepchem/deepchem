# DeepChem

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
* [DeepChem Publications](#deepchem-publications)
* [Examples](/examples)
* [About Us](#about-us)

## Requirements
* [openbabel](http://openbabel.org/wiki/Main_Page)
* [pandas](http://pandas.pydata.org/)
* [rdkit](http://www.rdkit.org/docs/Install.html)
* [boost](http://www.boost.org/)
* [joblib](https://pypi.python.org/pypi/joblib)
* [sklearn](https://github.com/scikit-learn/scikit-learn.git)
* [numpy](https://store.continuum.io/cshop/anaconda/)
* [keras](http://keras.io)
* [mdtraj](http://mdtraj.org/)
* [tensorflow](https://www.tensorflow.org/)

## Installation

Installation from source is the only currently supported format. ```deepchem``` currently supports both Python 2.7 and Python 3.5, but is not supported on any OS'es except 64 bit linux. Please make sure you follow the directions below precisely. While you may already have system versions of some of these packages, there is no guarantee that `deepchem` will work with alternate versions than those specified below.

1. Download the **64-bit** Python 2.7 or Python 3.5 versions of Anaconda for linux [here](https://www.continuum.io/downloads#_unix). 
   
   Follow the [installation instructions](http://docs.continuum.io/anaconda/install#linux-install)

2. `openbabel`
   ```bash
   conda install -c omnia openbabel=2.4.0
   ``` 

3. `rdkit`
   ```bash
   conda install -c omnia rdkit
   ```

4. `joblib`
   ```bash
   conda install joblib 
   ```

5. `keras`
   ```bash
   pip install keras
   ```
   `deepchem` only supports the `tensorflow` (default) backend for keras.
   
6. `mdtraj`
   ```bash
   conda install -c omnia mdtraj
   ```

7. `tensorflow`: Installing `tensorflow` on older versions of Linux (which
    have glibc < 2.17) can be very challenging. For these older Linux versions,
    contact your local sysadmin to work out a custom installation. If your
    version of Linux is recent, then the following command will work:
    ```
    conda install -c https://conda.anaconda.org/jjhelmus tensorflow
    ```

8. `deepchem`: Clone the `deepchem` github repo:
   ```bash
   git clone https://github.com/deepchem/deepchem.git
   ```
   `cd` into the `deepchem` directory and execute
   ```bash
   python setup.py install
   ```

9. To run test suite, install `nosetests`:
   ```bash
   pip install nose
   ```
   Make sure that the correct version of `nosetests` is active by running
   ```bash
   which nosetests 
   ```
   You might need to uninstall a system install of `nosetests` if
   there is a conflict.

10. If installation has been successful, all tests in test suite should pass:
    ```bash
    nosetests -v deepchem --nologcapture 
    ```
    Note that the full test-suite uses up a fair amount of memory. 
    Try running tests for one submodule at a time if memory proves an issue.

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
|           |Multitask network   |0.775              |0.634              |
|           |robust MT-NN        |0.803              |0.632              |
|           |graph convolution   |0.708              |0.594              |
|toxcast    |logistic regression |0.721              |0.575              |
|           |Multitask network   |0.830              |0.678              |
|           |robust MT-NN        |0.825              |0.680              |
|           |graph convolution   |0.821              |0.720              |

Random splitting

|Dataset    |Model               |Train score/ROC-AUC|Valid score/ROC-AUC|
|-----------|--------------------|-------------------|-------------------|
|tox21      |logistic regression |0.903              |0.735              |
|           |Multitask network   |0.856              |0.783              |
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
|           |Multitask network   |0.777        	     |0.655              |
|           |robust MT-NN        |0.804              |0.630              |
|           |graph convolution   |0.705        	     |0.618              |
|toxcast    |logistic regression |0.725        	     |0.586              |
|           |Multitask network   |0.836        	     |0.684              |
|           |robust MT-NN        |0.822              |0.681              |
|           |graph convolution   |0.820        	     |0.717              |

Scaffold splitting

|Dataset    |Model               |Train score/ROC-AUC|Valid score/ROC-AUC|
|-----------|--------------------|-------------------|-------------------|
|tox21      |logistic regression |0.900              |0.650              |
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
|           |Multitask network   |0.776              |0.557              |
|           |robust MT-NN        |0.797              |0.560              |
|           |graph convolution   |0.722              |0.583              |
|toxcast    |logistic regression |0.716              |0.492              |
|           |Multitask network   |0.828              |0.617              |
|           |robust MT-NN        |0.830              |0.614              |
|           |graph convolution   |0.832              |0.638              |

* Regression

|Dataset         |Model               |Splitting   |Train score/R2|Valid score/R2|
|----------------|--------------------|------------|--------------|--------------|
|delaney         |MT-NN regression    |Index       |0.773         |0.574         |
|                |graphconv regression|Index       |0.991         |0.825         |
|                |MT-NN regression    |Random      |0.769         |0.591         |
|                |graphconv regression|Random      |0.996         |0.873         |
|                |MT-NN regression    |Scaffold    |0.782         |0.426         |
|                |graphconv regression|Scaffold    |0.994         |0.606         |
|nci             |MT-NN regression    |Index       |0.171         |0.062         |
|                |graphconv regression|Index       |0.123         |0.048         |
|                |MT-NN regression    |Random      |0.168         |0.085         |
|                |graphconv regression|Random      |0.117         |0.076         |
|                |MT-NN regression    |Scaffold    |0.180         |0.052         |
|                |graphconv regression|Scaffold    |0.131         |0.046         |
|pdbbind(core)   |MT-NN regression    |Random      |0.973         |0.494         |
|pdbbind(refined)|MT-NN regression    |Random      |0.987         |0.503         |
|pdbbind(full)   |MT-NN regression    |Random      |0.983         |0.528         |
|kaggle          |MT-NN regression    |User-defined|0.748         |0.452         |

* General features

Number of tasks and examples in the datasets

|Dataset         |N(tasks)	|N(samples) |
|----------------|-----------|-----------| 
|tox21           |12         |8014       |
|muv             |17         |93127      |
|pcba            |128        |439863     |
|sider           |27         |1427       |
|toxcast         |617        |8615       |
|delaney         |1          |1128       |
|kaggle          |15         |173065     |
|nci             |60         |19127      |
|pdbbind(core)   |1          |195        |
|pdbbind(refined)|1          |3706       |
|pdbbind(full)   |1          |11908      |



Time needed for benchmark test(~20h in total)

|Dataset         |Model               |Time(loading)/s |Time(running)/s|
|----------------|--------------------|----------------|---------------| 
|tox21           |logistic regression |30              |60             |
|                |Multitask network   |30              |60             |
|                |robust MT-NN        |30              |90             |
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
|                |graph convolution   |20              |50             |
|toxcast         |logistic regression |80              |2600           |
|                |Multitask network   |80              |2300           |
|                |robust MT-NN        |80              |4000           |
|                |graph convolution   |80              |900            |
|delaney         |MT-NN regression    |10              |40             |
|                |graphconv regression|10              |40             |
|nci             |MT-NN regression    |400             |1200           |
|                |graphconv regression|400             |2500           |
|pdbbind(core)   |MT-NN regression    |0(featurized)   |30             |
|pdbbind(refined)|MT-NN regression    |0(featurized)   |40             |
|pdbbind(full)   |MT-NN regression    |0(featurized)   |60             |
|kaggle          |MT-NN regression    |2200            |3200           |


## Contributing to DeepChem

We actively encourage community contributions to DeepChem. The first place to start getting involved is by running our examples locally. Afterwards, we encourage contributors to give a shot to improving our documentation. While we take effort to provide good docs, there's plenty of room for improvement. All docs are hosted on Github, either in this `README.md` file, or in the `docs/` directory.

Once you've got a sense of how the package works, we encourage the use of Github issues to discuss more complex changes,  raise requests for new features or propose changes to the global architecture of DeepChem. Once consensus is reached on the issue, please submit a PR with proposed modifications. All contributed code to DeepChem will be reviewed by a member of the DeepChem team, so please make sure your code style and documentation style match our guidelines!

### Code Style Guidelines
DeepChem broadly follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). In terms of practical changes, the biggest effect is that all code uses 2-space indents instead of 4-space indents. We encourage new contributors to make use of [pylint](https://www.pylint.org/) with the following command
```
pylint --disable=invalid-name --indent-string "  " --extension-pkg-whitelist=numpy [file.py]
```
Aim for a score of at least 8/10 on contributed files.

### Documentation Style Guidelines
DeepChem uses [NumPy style documentation](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt). Please follow these conventions when documenting code, since we use [Sphinx+Napoleon](http://www.sphinx-doc.org/en/stable/ext/napoleon.html) to automatically generate docs on [deepchem.io](deepchem.io). 

## DeepChem Publications
1. [Computational Modeling of β-secretase 1 (BACE-1) Inhibitors using
Ligand Based
Approaches](http://pubs.acs.org/doi/abs/10.1021/acs.jcim.6b00290)
1. [Low Data Drug Discovery with One-shot Learning](https://arxiv.org/abs/1611.03199)

## About Us
DeepChem is a package by the [Pande group](https://pande.stanford.edu/) at Stanford. DeepChem was originally created by [Bharath Ramsundar](http://rbharath.github.io/), and has grown through the contributions of a number of undergraduate, graduate, and postdoctoral researchers working with the Pande lab.

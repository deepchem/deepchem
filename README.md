deep_chem
=============

Deep Learning Toolchain for Cheminformatics and Protein Analysis

Requirements
------------
* [rdkit](http://www.rdkit.org/docs/Install.html)
* [sklearn](https://github.com/scikit-learn/scikit-learn.git)
* [numpy](https://store.continuum.io/cshop/anaconda/)
* [keras](keras.io)
* [vs_utils] (https://github.com/pandegroup/vs-utils.git)

#Linux Installation

Deep_chem currently requires Python 2.7

##Anaconda 2.7
Download the **Python 2.7** version of Anaconda for linux (32 or 64 bit depending on system) here(www.continuum.io/downloads#_unix).

Follow the installation instructions(http://docs.continuum.io/anaconda/install#linux-install)

##Openbabel
```bash
conda install -c https://conda.anaconda.org/omnia openbabel
```  

Follow the onscreen installation instructions

##Rdkit
```bash
conda install -c https://conda.anaconda.org/omnia rdkit
```

Follow the onscreen installation instructions

##Keras
Clone the keras git repository
```bash
git clone https://github.com/fchollet/keras.git
```

Cd into the keras directory and execute
```bash
python setup.py install
```

##vs_utils
Clone the vs_utils git repository
```bash
git clone https://github.com/pandegroup/vs-utils.git
```

Cd into the vs-utils directory and execute
```bash
python setup.py develop
```

##deep_chem
Clone the deep_chem git repository
```bash
git clone https://github.com/pandegroup/deep-learning.git
```

Cd into the deep-learning directory and execute 
```bash
python setup.py develop
```

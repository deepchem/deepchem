MoleculeNet
===========
The DeepChem library is packaged alongside the MoleculeNet suite of datasets. One of the most important parts of machine learning applications is finding a suitable dataset. The MoleculeNet suite has curated a whole range of datasets and loaded them into DeepChem :code:`dc.data.Dataset` objects for convenience.

Contributing a new dataset to MoleculeNet
-----------------------------------------

If you are proposing a new dataset to be included in the MoleculeNet benchmarking suite, 
please follow the instructions below. Please review the `datasets already available in MolNet <http://moleculenet.ai/datasets-1>`_ before contributing.

0. Read the `Contribution guidelines <https://github.com/deepchem/deepchem/blob/master/CONTRIBUTING.md>`_.

1. Open an `issue <https://github.com/deepchem/deepchem/issues>`_ to discuss the dataset you want to add to MolNet.

2. Implement a function in the `deepchem.molnet.load_function <https://github.com/deepchem/deepchem/tree/master/deepchem/molnet/load_function>`_ module following the template function `deepchem.molnet.load_function.load_mydataset <https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/load_mydataset.py>`_.

3. Add your load function to `deepchem.molnet.__init__.py <https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/__init__.py>`_ for easy importing.

4. Prepare your dataset as a .tar.gz or .zip file. Accepted filetypes include CSV, JSON, and SDF.

5. Ask a member of the technical steering committee to add your .tar.gz or .zip file to the DeepChem AWS bucket. Modify your load function to pull down the dataset from AWS.

6. Submit a [WIR] PR (Work in progress pull request) following the PR `template <https://github.com/deepchem/deepchem/blob/master/docs/molnet_pr_template.md>`_.  

Load Dataset Template
---------------------

.. autofunction:: deepchem.molnet.load_function.load_mydataset

BACE Dataset
------------

.. autofunction:: deepchem.molnet.load_bace_classification

.. autofunction:: deepchem.molnet.load_bace_regression

BBBC Datasets
-------------

.. autofunction:: deepchem.molnet.load_bbbc001

.. autofunction:: deepchem.molnet.load_bbbc002

BBBP Datasets
-------------
BBBP stands for Blood-Brain-Barrier Penetration

.. autofunction:: deepchem.molnet.load_bbbp

Cell Counting Datasets
----------------------

.. autofunction:: deepchem.molnet.load_cell_counting

Chembl Datasets
---------------

.. autofunction:: deepchem.molnet.load_chembl

Chembl25 Datasets
---------------

.. autofunction:: deepchem.molnet.load_chembl25

Clearance Datasets
------------------

.. autofunction:: deepchem.molnet.load_clearance

Clintox Datasets
----------------

.. autofunction:: deepchem.molnet.load_clintox

Delaney Datasets
----------------

.. autofunction:: deepchem.molnet.load_delaney

Factors Datasets
----------------

.. autofunction:: deepchem.molnet.load_factors

HIV Datasets
------------

.. autofunction:: deepchem.molnet.load_hiv

HOPV Datasets
-------------
HOPV stands for the Harvard Organic Photovoltaic Dataset.

.. autofunction:: deepchem.molnet.load_hopv

HPPB Datasets
-------------

.. autofunction:: deepchem.molnet.load_hppb


KAGGLE Datasets
---------------

.. autofunction:: deepchem.molnet.load_kaggle

Kinase Datasets
---------------

.. autofunction:: deepchem.molnet.load_kinase


Lipo Datasets
-------------

.. autofunction:: deepchem.molnet.load_lipo

MUV Datasets
------------

.. autofunction:: deepchem.molnet.load_muv

NCI Datasets
------------

.. autofunction:: deepchem.molnet.load_nci

PCBA Datasets
-------------

.. autofunction:: deepchem.molnet.load_pcba

PDBBIND Datasets
----------------

.. autofunction:: deepchem.molnet.load_pdbbind

PPB Datasets
------------

.. autofunction:: deepchem.molnet.load_ppb

QM7 Datasets
------------

.. autofunction:: deepchem.molnet.load_qm7

.. autofunction:: deepchem.molnet.load_qm7_from_mat

.. autofunction:: deepchem.molnet.load_qm7b_from_mat

QM8 Datasets
------------

.. autofunction:: deepchem.molnet.load_qm8

QM9 Datasets
------------

.. autofunction:: deepchem.molnet.load_qm9


SAMPL Datasets
--------------

.. autofunction:: deepchem.molnet.load_sampl


SIDER Datasets
--------------

.. autofunction:: deepchem.molnet.load_sider

SWEETLEAD Datasets
------------------

.. autofunction:: deepchem.molnet.load_sweetlead

Thermosol Datasets
------------------

.. autofunction:: deepchem.molnet.load_thermosol


Tox21 Datasets
--------------

.. autofunction:: deepchem.molnet.load_tox21

Toxcast Datasets
----------------

.. autofunction:: deepchem.molnet.load_toxcast

USPTO Datasets
--------------

.. autofunction:: deepchem.molnet.load_uspto

UV Datasets
-----------

.. autofunction:: deepchem.molnet.load_uv

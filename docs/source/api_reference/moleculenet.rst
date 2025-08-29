MoleculeNet
===========
The DeepChem library is packaged alongside the MoleculeNet suite of datasets.
One of the most important parts of machine learning applications is finding a suitable dataset.
The MoleculeNet suite has curated a whole range of datasets and loaded them into DeepChem
:code:`dc.data.Dataset` objects for convenience.

.. include:: moleculenet_cheatsheet.rst

Contributing a new dataset to MoleculeNet
-----------------------------------------

If you are proposing a new dataset to be included in the
MoleculeNet benchmarking suite, please follow the instructions below.
Please review the `datasets already available in MolNet`_ before contributing.

0. Read the `Contribution guidelines`_.

1. Open an `issue`_ to discuss the dataset you want to add to MolNet.

2. Write a `DatasetLoader` class that inherits from `deepchem.molnet.load_function.molnet_loader._MolnetLoader`_ and implements a `create_dataset` method. See the `_QM9Loader`_ for a simple example.

3. Write a `load_dataset` function that documents the dataset and add your load function to `deepchem.molnet.__init__.py`_ for easy importing.

4. Prepare your dataset as a .tar.gz or .zip file. Accepted filetypes include CSV, JSON, and SDF.

5. Ask a member of the technical steering committee to add your .tar.gz or .zip file to the DeepChem AWS bucket. Modify your load function to pull down the dataset from AWS.

6. Add documentation for your loader to the `MoleculeNet docs`_.

7. Submit a [WIP] PR (Work in progress pull request) following the PR `template`_.

Example Usage
-------------
Below is an example of how to load a MoleculeNet dataset and featurizer. This approach will work for any dataset in MoleculeNet by changing the load function and featurizer. For more details on the featurizers, see the `Featurizers` section.
::


    import deepchem as dc
    from deepchem.feat.molecule_featurizers import MolGraphConvFeaturizer

    featurizer = MolGraphConvFeaturizer(use_edges=True)
    dataset_dc = dc.molnet.load_qm9(featurizer=featurizer)
    tasks, dataset, transformers = dataset_dc
    train, valid, test = dataset

    x,y,w,ids = train.X, train.y, train.w, train.ids


Note that the "w" matrix represents the weight of each sample. Some assays may have missing values, in which case the weight is 0. Otherwise, the weight is 1.


Additionally, the environment variable ``DEEPCHEM_DATA_DIR`` can be set like ``os.environ['DEEPCHEM_DATA_DIR'] = path/to/store/featurized/dataset``. When the ``DEEPCHEM_DATA_DIR`` environment variable is set, molnet loader stores the featurized dataset in the specified directory and when the dataset has to be reloaded the next time, it will be fetched from the data directory directly rather than featurizing the raw dataset from scratch.

BACE Dataset
------------

.. autofunction:: deepchem.molnet.load_bace_classification

.. autofunction:: deepchem.molnet.load_bace_regression

BBBC Datasets
-------------

.. autofunction:: deepchem.molnet.load_bbbc001

.. autofunction:: deepchem.molnet.load_bbbc002

.. autofunction:: deepchem.molnet.load_bbbc003

.. autofunction:: deepchem.molnet.load_bbbc004

.. autofunction:: deepchem.molnet.load_bbbc005

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
-----------------

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

Freesolv Dataset
----------------------

.. autofunction:: deepchem.molnet.load_freesolv

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

Materials Datasets
------------------
Materials datasets include inorganic crystal structures, chemical
compositions, and target properties like formation energies and band
gaps. Machine learning problems in materials science commonly include
predicting the value of a continuous (regression) or categorical
(classification) property of a material based on its chemical composition
or crystal structure. "Inverse design" is also of great interest, in which
ML methods generate crystal structures that have a desired property.
Other areas where ML is applicable in materials include: discovering new
or modified phenomenological models that describe material behavior

.. autofunction:: deepchem.molnet.load_bandgap
.. autofunction:: deepchem.molnet.load_perovskite
.. autofunction:: deepchem.molnet.load_mp_formation_energy
.. autofunction:: deepchem.molnet.load_mp_metallicity

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

QM8 Datasets
------------

.. autofunction:: deepchem.molnet.load_qm8

QM9 Datasets
------------

A bug was reported in the issue `https://github.com/deepchem/deepchem/issues/4413` in the previously 
included SDF files for the QM9 dataset, where some molecules incorrectly carried formal charges, 
despite QM9 molecules being charge-neutral. To address this, we now use the original QM9 XYZ files 
and convert them to SDF format using Open Babel, which preserves correct charge information. The updated
SDF file is uploaded to the Deepchem S3 bucket.

.. _`XYZ files`: https://doi.org/10.6084/m9.figshare.978904_D12

.. _`updated QM9 file`: https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/qm9.tar.gz

.. autofunction:: deepchem.molnet.load_qm9


SAMPL Datasets
--------------

.. autofunction:: deepchem.molnet.load_sampl


SIDER Datasets
--------------

.. autofunction:: deepchem.molnet.load_sider


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


.. _`datasets already available in MolNet`: https://moleculenet.org/datasets-1
.. _`Contribution guidelines`: https://github.com/deepchem/deepchem/blob/master/CONTRIBUTING.md
.. _`issue`: https://github.com/deepchem/deepchem/issues
.. _`_QM9Loader`: https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/qm9_datasets.py
.. _`deepchem.molnet.load_function.molnet_loader._MolnetLoader`: https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/molnet_loader.py#L82
.. _`deepchem.molnet.load_function`: https://github.com/deepchem/deepchem/tree/master/deepchem/molnet/load_function
.. _`deepchem.molnet.load_function.load_dataset_template`: https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/load_dataset_template.py
.. _`deepchem.molnet.defaults`: https://github.com/deepchem/deepchem/tree/master/deepchem/molnet/defaults.py
.. _`deepchem.molnet.__init__.py`: https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/__init__.py
.. _`MoleculeNet docs`: https://github.com/deepchem/deepchem/blob/master/docs/source/api_reference/moleculenet.rst
.. _`template`: https://github.com/deepchem/deepchem/blob/master/.github/MOLNET_PR_TEMPLATE.md

ZINC15 Datasets
---------------

.. autofunction:: deepchem.molnet.load_zinc15

Platinum Adsorption Dataset
---------------------------

.. autofunction:: deepchem.molnet.load_Platinum_Adsorption

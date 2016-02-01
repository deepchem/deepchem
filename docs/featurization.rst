.. _featurization:

Featurization
=============

Input Format
------------
Accepted input formats include csv, pkl.gz, and sdf file. For simplicity,
let's assume we deal with a csv input. In order to build models, we expect
the following columns to have entries for each row in the csv file.

1. ``smiles_field``: A column


Data Featurization
------------------


Most machine learning algorithms require that input data form vectors.
However, input data for drug-discovery datasets routinely come in the
format of lists of molecules and associated experimental readouts. To
transform lists of molecules into vectors, we need to use ``deechem``'s
featurization class ``DataFeaturizer``

.. autosummary::
    :toctree: _featurization/

    deepchem.featurizers

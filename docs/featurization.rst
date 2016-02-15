.. _featurization:

Featurization
=============

Data Featurization
------------------

Most machine learning algorithms require that input data form vectors.
However, input data for drug-discovery datasets routinely come in the
format of lists of molecules and associated experimental readouts. To
transform lists of molecules into vectors, we need to use the ``deechem``
featurization class ``DataFeaturizer``. Instances of this class must be
passed a ``Featurizer`` object. ``deepchem`` provides a number of
different subclasses of ``Featurizer`` for convenience:

Featurizers.

.. autosummary::
    :toctree: _featurization/

    deepchem.featurizers.basic
    deepchem.featurizers.fingerprints
    deepchem.featurizers.nnscore
    deepchem.featurizers.coulomb_matrices

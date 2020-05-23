Featurizers
===========

DeepChem contains an extensive collection of featurizers. If you
haven't run into this terminology before, a "featurizer" is chunk of
code which transforms raw input data into a processed form suitable
for machine learning. Machine learning methods often need data to be
pre-chewed for them to process. Think of this like a mama penguin
chewing up food so the baby penguin can digest it easily.


Now if you've watched a few introductory deep learning lectures, you
might ask, why do we need something like a featurizer? Isn't part of
the promise of deep learning that we can learn patterns directly from
raw data?

Unfortunately it turns out that deep learning techniques need
featurizers just like normal machine learning methods do. Arguably,
they are less dependent on sophisticated featurizers and more capable
of learning sophisticated patterns from simpler data. But
nevertheless, deep learning systems can't simply chew up raw files.
For this reason, :code:`deepchem` provides an extensive collection of
featurization methods which we will review on this page.

Featurizer
----------

The :code:`dc.feat.Featurizer` class is the abstract parent class for all featurizers.

.. autoclass:: deepchem.feat.Featurizer
  :members:

MolecularFeaturizer
-------------------

The :code:`dc.feat.MolecularFeaturizer` class is the abstract parent class for all featurizers that work with small molecule data. 

.. autoclass:: deepchem.feat.MolecularFeaturizer
  :members:

ConvMolFeaturizer
^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.ConvMolFeaturizer
  :members:

WeaveFeaturizer
^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.WeaveFeaturizer
  :members:

CircularFingerprint
^^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.CircularFingerprint
  :members:

RDKitDescriptors
^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.RDKitDescriptors
  :members:

CoulombMatrix
^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.CoulombMatrix
  :members:

CoulombMatrixEig
^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.CoulombMatrixEig
  :members:


ReactionFeaturizer
------------------

The :code:`dc.feat.ReactionFeaturizer` class is the abstract parent class for all featurizers that work with chemical reaction data.

.. autoclass:: deepchem.feat.ReactionFeaturizer
  :members:

MolecularComplexFeaturizer
--------------------------

The :code:`dc.feat.MolecularComplexFeaturizer` class is the abstract parent class for all featurizers that work with three dimensional molecular complexes. 


.. autoclass:: deepchem.feat.ComplexFeaturizer
  :members:

RdkitGridFeaturizer
^^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.RdkitGridFeaturizer
  :members:

UserDefinedFeaturizer
---------------------

.. autoclass:: deepchem.feat.UserDefinedFeaturizer
  :members:

BPSymmetryFunctionInput
-----------------------

.. autoclass:: deepchem.feat.BPSymmetryFunctionInput
  :members:

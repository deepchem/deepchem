Transformers
============
DeepChem :code:`dc.trans.Transformer` objects are another core
building block of DeepChem programs. Often times, machine learning
systems are very delicate. They need their inputs and outputs to fit
within a pre-specified range or follow a clean mathematical
distribution. Real data of course is wild and hard to control. What do
you do if you have a crazy dataset and need to bring its statistics to
heel? Fear not for you have :code:`Transformer` objects.

.. contents:: Contents
    :local:

General Transformers
--------------------

NormalizationTransformer
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.trans.NormalizationTransformer
  :members:
  :inherited-members:

MinMaxTransformer
^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.trans.MinMaxTransformer
  :members:
  :inherited-members:

ClippingTransformer
^^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.trans.ClippingTransformer
  :members:
  :inherited-members:

LogTransformer
^^^^^^^^^^^^^^

.. autoclass:: deepchem.trans.LogTransformer
  :members:
  :inherited-members:

CDFTransformer
^^^^^^^^^^^^^^

.. autoclass:: deepchem.trans.CDFTransformer
  :members:
  :inherited-members:

PowerTransformer
^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.trans.PowerTransformer
  :members:
  :inherited-members:

BalancingTransformer
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.trans.BalancingTransformer
  :members:
  :inherited-members:

DuplicateBalancingTransformer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.trans.DuplicateBalancingTransformer
  :members:
  :inherited-members:

ImageTransformer
^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.trans.ImageTransformer
  :members:
  :inherited-members:

FeaturizationTransformer
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.trans.FeaturizationTransformer
  :members:
  :inherited-members:

Specified Usecase Transformers
------------------------------

CoulombFitTransformer
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.trans.CoulombFitTransformer
  :members:
  :inherited-members:

IRVTransformer
^^^^^^^^^^^^^^

.. autoclass:: deepchem.trans.IRVTransformer
  :members:
  :inherited-members:

DAGTransformer
^^^^^^^^^^^^^^

.. autoclass:: deepchem.trans.DAGTransformer
  :members:
  :inherited-members:

ANITransformer
^^^^^^^^^^^^^^

.. autoclass:: deepchem.trans.ANITransformer
  :members:
  :inherited-members:

Base Transformer (for develop)
-------------------------------

The :code:`dc.trans.Transformer` class is the abstract parent class
for all transformers. This class should never be directly initialized,
but contains a number of useful method implementations.

.. autoclass:: deepchem.trans.Transformer
  :members:

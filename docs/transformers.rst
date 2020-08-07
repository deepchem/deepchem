Transformers
============
DeepChem :code:`dc.trans.Transformer` objects are another core
building block of DeepChem programs. Often times, machine learning
systems are very delicate. They need their inputs and outputs to fit
within a pre-specified range or follow a clean mathematical
distribution. Real data of course is wild and hard to control. What do
you do if you have a crazy dataset and need to bring its statistics to
heel? Fear not for you have :code:`Transformer` objects.

Transformer
-----------
The :code:`dc.trans.Transformer` class is the abstract parent class
for all transformers. This class should never be directly initialized,
but contains a number of useful method implementations.

.. autoclass:: deepchem.trans.Transformer
  :members:

MinMaxTransformer
-----------------

.. autoclass:: deepchem.trans.MinMaxTransformer
  :members:

NormalizationTransformer
------------------------

.. autoclass:: deepchem.trans.NormalizationTransformer
  :members:

ClippingTransformer
-------------------

.. autoclass:: deepchem.trans.ClippingTransformer
  :members:

LogTransformer
--------------

.. autoclass:: deepchem.trans.LogTransformer
  :members:

BalancingTransformer
--------------------

.. autoclass:: deepchem.trans.BalancingTransformer
  :members:

DuplicateBalancingTransformer
-----------------------------

.. autoclass:: deepchem.trans.DuplicateBalancingTransformer
  :members:

CDFTransformer
--------------

.. autoclass:: deepchem.trans.CDFTransformer
  :members:

PowerTransformer
----------------

.. autoclass:: deepchem.trans.PowerTransformer
  :members:

CoulombFitTransformer
---------------------

.. autoclass:: deepchem.trans.CoulombFitTransformer
  :members:

IRVTransformer
--------------

.. autoclass:: deepchem.trans.IRVTransformer
  :members:

DAGTransformer
--------------

.. autoclass:: deepchem.trans.DAGTransformer
  :members:

ImageTransformer
----------------

.. autoclass:: deepchem.trans.ImageTransformer
  :members:

ANITransformer
--------------

.. autoclass:: deepchem.trans.ANITransformer
  :members:

FeaturizationTransformer
------------------------

.. autoclass:: deepchem.trans.FeaturizationTransformer
  :members:

DataTransforms
--------------

.. autoclass:: deepchem.trans.DataTransforms
  :members:

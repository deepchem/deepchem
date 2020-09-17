Splitters
=========
DeepChem :code:`dc.splits.Splitter` objects are a tool to meaningfully
split DeepChem datasets for machine learning testing. The core idea is
that when evaluating a machine learning model, it's useful to creating
training, validation and test splits of your source data. The training
split is used to train models, the validatation is used to benchmark
different model architectures. The test is ideally held out till the
very end when it's used to gauge a final estimate of the model's
performance.

The :code:`dc.splits` module contains a collection of scientifically
aware splitters. In many cases, we want to evaluate scientific deep
learning models more rigorously than standard deep models since we're
looking for the ability to generalize to new domains. Some of the
implemented splitters here may help.

Splitter
--------
The :code:`dc.splits.Splitter` class is the abstract parent class for
all splitters. This class should never be directly instantiated.

.. autoclass:: deepchem.splits.Splitter
  :members:

RandomSplitter
--------------

.. autoclass:: deepchem.splits.RandomSplitter
  :members:

IndexSplitter
-------------

.. autoclass:: deepchem.splits.IndexSplitter
  :members:

SpecifiedSplitter
--------------

.. autoclass:: deepchem.splits.SpecifiedSplitter
  :members:


RandomGroupSplitter
-------------------

.. autoclass:: deepchem.splits.RandomGroupSplitter
  :members:

RandomStratifiedSplitter
------------------------

.. autoclass:: deepchem.splits.RandomStratifiedSplitter
  :members:

SingletaskStratifiedSplitter
----------------------------

.. autoclass:: deepchem.splits.SingletaskStratifiedSplitter
  :members:

MolecularWeightSplitter
-----------------------

.. autoclass:: deepchem.splits.MolecularWeightSplitter
  :members:

MaxMinSplitter
--------------

.. autoclass:: deepchem.splits.MaxMinSplitter
  :members:

ButinaSplitter
--------------

.. autoclass:: deepchem.splits.ButinaSplitter
  :members:

ScaffoldSplitter
----------------

.. autoclass:: deepchem.splits.ScaffoldSplitter
  :members:

FingeprintSplitter
------------------

.. autoclass:: deepchem.splits.FingerprintSplitter
  :members:


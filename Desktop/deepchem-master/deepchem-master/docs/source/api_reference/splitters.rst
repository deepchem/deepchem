Splitters
=========
DeepChem :code:`dc.splits.Splitter` objects are a tool to meaningfully
split DeepChem datasets for machine learning testing. The core idea is
that when evaluating a machine learning model, it's useful to creating
training, validation and test splits of your source data. The training
split is used to train models, the validation is used to benchmark
different model architectures. The test is ideally held out till the
very end when it's used to gauge a final estimate of the model's
performance.

The :code:`dc.splits` module contains a collection of scientifically
aware splitters. In many cases, we want to evaluate scientific deep
learning models more rigorously than standard deep models since we're
looking for the ability to generalize to new domains. Some of the
implemented splitters here may help.

.. contents:: Contents
    :local:

General Splitters
-----------------

RandomSplitter
^^^^^^^^^^^^^^

.. autoclass:: deepchem.splits.RandomSplitter
  :members:
  :inherited-members:
  :exclude-members: __init__


RandomGroupSplitter
^^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.splits.RandomGroupSplitter
  :members:
  :inherited-members:

RandomStratifiedSplitter
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.splits.RandomStratifiedSplitter
  :members:
  :inherited-members:
  :exclude-members: __init__

SingletaskStratifiedSplitter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.splits.SingletaskStratifiedSplitter
  :members:
  :inherited-members:

IndexSplitter
^^^^^^^^^^^^^

.. autoclass:: deepchem.splits.IndexSplitter
  :members:
  :inherited-members:
  :exclude-members: __init__

SpecifiedSplitter
^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.splits.SpecifiedSplitter
  :members:
  :inherited-members:

TaskSplitter
^^^^^^^^^^^^

.. autoclass:: deepchem.splits.TaskSplitter
  :members:
  :inherited-members:

Time-based Splitting
^^^^^^^^^^^^^^^^^^^^
Time-based Splitting is a technique used to divide datasets according to their chronological order.

In this method, the earlier data points are used for the training of the model, while the later data points are reserved for validation and testing. 
This allows the model to be evaluated on data that occurs later in time, simulating real-world scenarios where predictions are made on future data.

By maintaining the chronological order of the dataset, time-based splitting helps avoid data leakage, since information from future data points  is not used during training.

This technique is especially useful for time-dependent datasets and is pivotal for  reproducing benchmark results such as those in MoleculeNet.



Molecule Splitters
------------------

ScaffoldSplitter
^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.splits.ScaffoldSplitter
  :members:
  :inherited-members:
  :exclude-members: __init__

MolecularWeightSplitter
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.splits.MolecularWeightSplitter
  :members:
  :inherited-members:
  :exclude-members: __init__

MaxMinSplitter
^^^^^^^^^^^^^^

.. autoclass:: deepchem.splits.MaxMinSplitter
  :members:
  :inherited-members:
  :exclude-members: __init__

ButinaSplitter
^^^^^^^^^^^^^^

.. autoclass:: deepchem.splits.ButinaSplitter
  :members:
  :inherited-members:

FingerprintSplitter
^^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.splits.FingerprintSplitter
  :members:
  :inherited-members:
  :exclude-members: __init__

Base Splitter (for develop)
----------------------------

The :code:`dc.splits.Splitter` class is the abstract parent class for
all splitters. This class should never be directly instantiated.

.. autoclass:: deepchem.splits.Splitter
  :members:

Datasets
========

DeepChem :code:`dc.data.Dataset` objects are one of the core building blocks of DeepChem programs. :code:`Dataset` objects hold representations of data for machine learning and are widely used throughout DeepChem.

Dataset
-------
The :code:`dc.data.Dataset` class is the abstract parent clss for all datasets. This class should never be directly initialized, but contains a number of useful method implementations.

The goal of the :code:`Dataset` class is to be maximally interoperable with other common representations of machine learning datasets. For this reason we provide interconversion methods mapping from :code:`Dataset` objects to pandas dataframes, tensorflow Datasets, and PyTorch datasets.

.. autoclass:: deepchem.data.Dataset
  :members:

NumpyDataset
------------
The :code:`dc.data.NumpyDataset` class provides an in-memory implementation of the abstract :code:`Dataset` which stores its data in :code:`numpy.ndarray` objects.

.. autoclass:: deepchem.data.NumpyDataset
  :members:

DiskDataset
-----------
The :code:`dc.data.DiskDataset` class allows for the storage of larger
datasets on disk. Each :code:`DiskDataset` is associated with a
directory in which it writes its contents to disk. Note that a
:code:`DiskDataset` can be very large, so some of the utility methods
to access fields of a :code:`Dataset` can be prohibitively expensive.

.. autoclass:: deepchem.data.DiskDataset
  :members:

ImageDataset
------------
The :code:`dc.data.ImageDataset` class is optimized to allow for convenient processing of image based datasets.

.. autoclass:: deepchem.data.ImageDataset
  :members:


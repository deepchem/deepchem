Data
====

DeepChem :code:`dc.data` provides APIs for handling your data.

If your data is stored by the file like CSV and SDF, you can use the **Data Loaders**.
The Data Loaders read your data, convert them to features (ex: SMILES to ECFP) and save the features to Dataset class.
If your data is python objects like Numpy arrays or Pandas DataFrames, you can use the **Datasets** directly.

.. contents:: Contents
    :local:


Datasets
--------

DeepChem :code:`dc.data.Dataset` objects are one of the core building blocks of DeepChem programs.
:code:`Dataset` objects hold representations of data for machine learning and are widely used throughout DeepChem.

The goal of the :code:`Dataset` class is to be maximally interoperable
with other common representations of machine learning datasets. 
For this reason we provide interconversion methods mapping from :code:`Dataset` objects
to pandas DataFrames, TensorFlow Datasets, and PyTorch datasets.

NumpyDataset
^^^^^^^^^^^^
The :code:`dc.data.NumpyDataset` class provides an in-memory implementation of the abstract :code:`Dataset`
which stores its data in :code:`numpy.ndarray` objects.

.. autoclass:: deepchem.data.NumpyDataset
  :members:
  :inherited-members:

DiskDataset
^^^^^^^^^^^
The :code:`dc.data.DiskDataset` class allows for the storage of larger
datasets on disk. Each :code:`DiskDataset` is associated with a
directory in which it writes its contents to disk. Note that a
:code:`DiskDataset` can be very large, so some of the utility methods
to access fields of a :code:`Dataset` can be prohibitively expensive.

.. autoclass:: deepchem.data.DiskDataset
  :members:
  :inherited-members:

ImageDataset
^^^^^^^^^^^^
The :code:`dc.data.ImageDataset` class is optimized to allow
for convenient processing of image based datasets.

.. autoclass:: deepchem.data.ImageDataset
  :members:
  :inherited-members:


Data Loaders
------------

Processing large amounts of input data to construct a :code:`dc.data.Dataset` object can require some amount of hacking.
To simplify this process for you, you can use the :code:`dc.data.DataLoader` classes.
These classes provide utilities for you to load and process large amounts of data.

CSVLoader
^^^^^^^^^

.. autoclass:: deepchem.data.CSVLoader
  :members: __init__, create_dataset

UserCSVLoader
^^^^^^^^^^^^^

.. autoclass:: deepchem.data.UserCSVLoader
  :members: __init__, create_dataset

ImageLoader
^^^^^^^^^^^

.. autoclass:: deepchem.data.ImageLoader
  :members: __init__, create_dataset

JsonLoader
^^^^^^^^^^
JSON is a flexible file format that is human-readable, lightweight, 
and more compact than other open standard formats like XML. JSON files
are similar to python dictionaries of key-value pairs. All keys must
be strings, but values can be any of (string, number, object, array,
boolean, or null), so the format is more flexible than CSV. JSON is
used for describing structured data and to serialize objects. It is
conveniently used to read/write Pandas dataframes with the
`pandas.read_json` and `pandas.write_json` methods.

.. autoclass:: deepchem.data.JsonLoader
  :members: __init__, create_dataset

SDFLoader
^^^^^^^^^

.. autoclass:: deepchem.data.SDFLoader
  :members: __init__, create_dataset

FASTALoader
^^^^^^^^^^^

.. autoclass:: deepchem.data.FASTALoader
  :members: __init__, create_dataset

InMemoryLoader
^^^^^^^^^^^^^^
The :code:`dc.data.InMemoryLoader` is designed to facilitate the processing of large datasets
where you already hold the raw data in-memory (say in a pandas dataframe).

.. autoclass:: deepchem.data.InMemoryLoader
  :members: __init__, create_dataset


Data Classes
------------
DeepChem featurizers often transform members into "data classes". These are
classes that hold all the information needed to train a model on that data
point. Models then transform these into the tensors for training in their
:code:`default_generator` methods.

Graph Data
^^^^^^^^^^

These classes document the data classes for graph convolutions. 
We plan to simplify these classes (:code:`ConvMol`, :code:`MultiConvMol`, :code:`WeaveMol`)
into a joint data representation (:code:`GraphData`) for all graph convolutions in a future version of DeepChem,
so these APIs may not remain stable.

The graph convolution models which inherit :code:`KerasModel` depend on :code:`ConvMol`, :code:`MultiConvMol`, or :code:`WeaveMol`.
On the other hand, the graph convolution models which inherit :code:`TorchModel` depend on :code:`GraphData`.

.. autoclass:: deepchem.feat.mol_graphs.ConvMol
  :members:

.. autoclass:: deepchem.feat.mol_graphs.MultiConvMol
  :members:
  :undoc-members:

.. autoclass:: deepchem.feat.mol_graphs.WeaveMol
  :members:
  :undoc-members:

.. autoclass:: deepchem.feat.graph_data.GraphData
  :members:


Base Classes (for develop)
--------------------------

Dataset
^^^^^^^
The :code:`dc.data.Dataset` class is the abstract parent class for all
datasets. This class should never be directly initialized, but
contains a number of useful method implementations.

.. autoclass:: deepchem.data.Dataset
  :members:

DataLoader
^^^^^^^^^^

The :code:`dc.data.DataLoader` class is the abstract parent class for all
dataloaders. This class should never be directly initialized, but
contains a number of useful method implementations.

.. autoclass:: deepchem.data.DataLoader
  :members:

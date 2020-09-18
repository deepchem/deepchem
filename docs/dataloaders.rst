Data Loaders
============

Processing large amounts of input data to construct a :code:`dc.data.Dataset` object can require some amount of hacking. To simplify this process for you, you can use the :code:`dc.data.DataLoader` classes. These classes provide utilities for you to load and process large amounts of data.


DataLoader
----------

.. autoclass:: deepchem.data.DataLoader
  :members:

CSVLoader
^^^^^^^^^

.. autoclass:: deepchem.data.CSVLoader
  :members:

UserCSVLoader
^^^^^^^^^^^^^

.. autoclass:: deepchem.data.UserCSVLoader
  :members:

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
  :members:

FASTALoader
^^^^^^^^^^^

.. autoclass:: deepchem.data.FASTALoader
  :members:

ImageLoader
^^^^^^^^^^^

.. autoclass:: deepchem.data.ImageLoader
  :members:

SDFLoader
^^^^^^^^^

.. autoclass:: deepchem.data.SDFLoader
  :members:

InMemoryLoader
^^^^^^^^^^^^^^
The :code:`dc.data.InMemoryLoader` is designed to facilitate the processing of large datasets where you already hold the raw data in-memory (say in a pandas dataframe).

.. autoclass:: deepchem.data.InMemoryLoader
  :members:

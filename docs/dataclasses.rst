Data Classes
============
DeepChem featurizers often transform members into "data classes". These are
classes that hold all the information needed to train a model on that data
point. Models then transform these into the tensors for training in their
:code:`default_generator` methods.

Graph Convolutions
------------------

These classes document the data classes for graph convolutions. We plan to simplify these classes into a joint data representation for all graph convolutions in a future version of DeepChem, so these APIs may not remain stable.

.. autoclass:: deepchem.feat.mol_graphs.ConvMol
  :members:

.. autoclass:: deepchem.feat.mol_graphs.MultiConvMol
  :members:

.. autoclass:: deepchem.feat.mol_graphs.WeaveMol
  :members:

.. autoclass:: deepchem.feat.graph_data.GraphData
  :members:

.. autoclass:: deepchem.feat.graph_data.BatchGraphData
  :members:

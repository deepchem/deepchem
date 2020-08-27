Metalearning
============
One of the hardest challenges in scientific machine learning is lack of access of sufficient data. Sometimes experiments are slow and expensive and there's no easy way to gain access to more data. What do you do then? 

This module contains a collection of techniques for doing low data
learning. "Metalearning" traditionally refers to techniques for
"learning to learn" but here we take it to mean any technique which
proves effective for learning with low amounts of data.

MetaLearner
-----------
This is the abstract superclass for metalearning algorithms.

.. autoclass:: deepchem.metalearning.MetaLearner
  :members:

MAML
----

.. autoclass:: deepchem.metalearning.MAML
  :members:

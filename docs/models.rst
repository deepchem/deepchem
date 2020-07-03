Model Classes
=============

Model
-----

.. autoclass:: deepchem.models.Model
  :members:

SklearnModel
------------

.. autoclass:: deepchem.models.SklearnModel
  :members:

XGBoostModel
------------

.. autoclass:: deepchem.models.XGBoostModel
  :members:

KerasModel
----------
DeepChem extensively uses `Keras`_ to build powerful machine learning models.

Training loss and validation metrics can be automatically logged to `Weights & Biases`_ with the following commands::

  # Install wandb in shell
  pip install wandb

  # Login in shell (required only once)
  wandb login

  # Start a W&B run in your script (refer to docs for optional parameters)
  wandb.init(project="my project")

  # Set `wandb` arg when creating `KerasModel`
  model = KerasModel(…, wandb=True)

.. _`Keras`: https://keras.io/

.. _`Weights & Biases`: http://docs.wandb.com/

.. autoclass:: deepchem.models.KerasModel
  :members:

MultitaskRegressor
------------------

.. autoclass:: deepchem.models.MultitaskRegressor
  :members:

MultitaskFitTransformRegressor
------------------------------

.. autoclass:: deepchem.models.MultitaskClassifier
  :members:

MultitaskClassifier
-------------------

.. autoclass:: deepchem.models.MultitaskClassifier
  :members:

TensorflowMultitaskIRVClassifier
--------------------------------

.. autoclass:: deepchem.models.TensorflowMultitaskIRVClassifier
  :members:

RobustMultitaskClassifier
-------------------------

.. autoclass:: deepchem.models.RobustMultitaskClassifier
  :members:

RobustMultitaskRegressor
------------------------

.. autoclass:: deepchem.models.RobustMultitaskRegressor
  :members:

ProgressiveMultitaskClassifier
------------------------------

.. autoclass:: deepchem.models.ProgressiveMultitaskClassifier
  :members:

ProgressiveMultitaskRegressor
-----------------------------

.. autoclass:: deepchem.models.ProgressiveMultitaskRegressor
  :members:

WeaveModel
----------

.. autoclass:: deepchem.models.WeaveModel
  :members:

DTNNModel
---------

.. autoclass:: deepchem.models.DTNNModel
  :members:

DAGModel
--------

.. autoclass:: deepchem.models.DAGModel
  :members:

GraphConvModel
--------------

.. autoclass:: deepchem.models.GraphConvModel
  :members:

MPNNModel
---------

.. autoclass:: deepchem.models.MPNNModel
  :members:

ScScoreModel
------------

.. autoclass:: deepchem.models.ScScoreModel
  :members:

SeqToSeq
--------

.. autoclass:: deepchem.models.SeqToSeq
  :members:

GAN
---

.. autoclass:: deepchem.models.GAN
  :members:

WGAN
^^^^

.. autoclass:: deepchem.models.WGAN
  :members:

CNN
---

.. autoclass:: deepchem.models.CNN
  :members:

TextCNNModel
------------

.. autoclass:: deepchem.models.CNN
  :members:


AtomicConvModel
---------------

.. autoclass:: deepchem.models.AtomicConvModel
  :members:


Smiles2Vec
----------

.. autoclass:: deepchem.models.Smiles2Vec
  :members:

ChemCeption
-----------

.. autoclass:: deepchem.models.ChemCeption
  :members:

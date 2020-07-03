Model Classes
=============

DeepChem maintains an extensive collection of models for scientific applications.

Model Cheatsheet
----------------
If you're just getting started with DeepChem, you're probably interested in the
basics. The place to get started is this "model cheatsheet" that lists various
types of custom DeepChem models. Note that some wrappers like `SklearnModel`
and `XGBoostModel` which wrap external machine learning libraries are excluded,
but this table is otherwise complete.

As a note about how to read this table, each row describes what's needed to
invoke a given model. Some models must be applied with given `Transformer` or
`Featurizer` objects. Some models also have custom training methods. You can
read off what's needed to train the model from the table below.

+----------------------------------+------------+----------------+-----------------+----------------------------------------------------------------+---------------+
| Model                            | Type       | Input Type     | Transformations | Acceptable Featurizers                                         | Fit Method    |
+==================================+============+================+=================+================================================================+===============+
| `AtomicConvModel`                | Classifier/| Tuple          |                 | `ComplexNeighborListFragmentAtomicCoordinates`                 | `fit`         |
|                                  | Regressor  |                |                 |                                                                |               |
+----------------------------------+------------+----------------+-----------------+----------------------------------------------------------------+---------------+
| `ChemCeption`                    | Classifier/| Tensor of shape|                 | `SmilesToImage`                                                | `fit`         |
|                                  | Regressor  | `(N, M, c)`    |                 |                                                                |               |
+----------------------------------+------------+----------------+-----------------+----------------------------------------------------------------+---------------+
| `CNN`                            | Classifier/| Tensor of shape|                 |                                                                | `fit`         |
|                                  | Regressor  | `(N, c)` or    |                 |                                                                |               |
|                                  |            | `(N, M, c)` or |                 |                                                                |               |
|                                  |            | `(N, M, L, c)` |                 |                                                                |               |
+----------------------------------+------------+----------------+-----------------+----------------------------------------------------------------+---------------+
| `DTNNModel`                      | Classifier/| Matrix of      |                 | `CoulombMatrix`                                                | `fit`         |
|                                  | Regressor  | shape `(N, N)` |                 |                                                                |               |
+----------------------------------+------------+----------------+-----------------+----------------------------------------------------------------+---------------+
| `DAGModel`                       | Classifier/| `ConvMol`      | `DAGTransformer`| `ConvMolFeaturizer`                                            | `fit`         |
|                                  | Regressor  |                |                 |                                                                |               |
+----------------------------------+------------+----------------+-----------------+----------------------------------------------------------------+---------------+
| `GraphConvModel`                 | Classifier/| `ConvMol`      |                 | `ConvMolFeaturizer`                                            | `fit`         |
|                                  | Regressor  |                |                 |                                                                |               |
+----------------------------------+------------+----------------+-----------------+----------------------------------------------------------------+---------------+
| `MPNNModel`                      | Classifier/| `WeaveMol`     |                 | `WeaveFeaturizer`                                              | `fit`         |
|                                  | Regressor  |                |                 |                                                                |               |
+----------------------------------+------------+----------------+-----------------+----------------------------------------------------------------+---------------+
| `MultitaskClassifier`            | Classifier | Vector of      |                 | `CircularFingerprint`,                                         | `fit`         | 
|                                  |            | shape `(N,)`   |                 | `RDKitDescriptors`, `CoulombMatrixEig`, `RdkitGridFeaturizer`, |               |
|                                  |            |                |                 | `BindingPocketFeaturizer`,                                     |               |
|                                  |            |                |                 | `AdjacencyFingerprint`, `ElementPropertyFingerprint`,          |               |
+----------------------------------+------------+----------------+-----------------+----------------------------------------------------------------+---------------+
| `MultitaskRegressor`             | Classifier | Vector of      |                 | `CircularFingerprint`,                                         | `fit`         |
|                                  |            | shape `(N,)`   |                 | `RDKitDescriptors`, `CoulombMatrixEig`, `RdkitGridFeaturizer`, |               |
|                                  |            |                |                 | `BindingPocketFeaturizer`,                                     |               |
|                                  |            |                |                 | `AdjacencyFingerprint`, `ElementPropertyFingerprint`,          |               |
+----------------------------------+------------+----------------+-----------------+----------------------------------------------------------------+---------------+
| `MultitaskRegressor`             | Classifier | Vector of      |                 | `CircularFingerprint`,                                         | `fit`         |
|                                  |            | shape `(N,)`   |                 | `RDKitDescriptors`, `CoulombMatrixEig`, `RdkitGridFeaturizer`, |               |
|                                  |            |                |                 | `BindingPocketFeaturizer`,                                     |               |
|                                  |            |                |                 | `AdjacencyFingerprint`, `ElementPropertyFingerprint`,          |               |
+----------------------------------+------------+----------------+-----------------+----------------------------------------------------------------+---------------+
| `MultitaskFitTransformRegressor` | Classifier | Vector of      |                 | `CircularFingerprint`,                                         | `fit`         |
|                                  |            | shape `(N,)`   |                 | `RDKitDescriptors`, `CoulombMatrixEig`, `RdkitGridFeaturizer`, |               |
|                                  |            |                |                 | `BindingPocketFeaturizer`,                                     |               |
|                                  |            |                |                 | `AdjacencyFingerprint`, `ElementPropertyFingerprint`,          |               |
+----------------------------------+------------+----------------+-----------------+----------------------------------------------------------------+---------------+
| `MultitaskRVClassifier`          | Classifier | Vector of      | `IRVTransformer`| `CircularFingerprint`,                                         | `fit`         |
|                                  |            | shape `(N,)`   |                 | `RDKitDescriptors`, `CoulombMatrixEig`, `RdkitGridFeaturizer`, |               |
|                                  |            |                |                 | `BindingPocketFeaturizer`,                                     |               |
|                                  |            |                |                 | `AdjacencyFingerprint`, `ElementPropertyFingerprint`,          |               |
+----------------------------------+------------+----------------+-----------------+----------------------------------------------------------------+---------------+
| `ProgressiveMultitaskClassifier` | Classifier | Vector of      |                 | `CircularFingerprint`,                                         | `fit`         |
|                                  |            | shape `(N,)`   |                 | `RDKitDescriptors`, `CoulombMatrixEig`, `RdkitGridFeaturizer`, |               |
|                                  |            |                |                 | `BindingPocketFeaturizer`,                                     |               |
|                                  |            |                |                 | `AdjacencyFingerprint`, `ElementPropertyFingerprint`,          |               |
+----------------------------------+------------+----------------+-----------------+----------------------------------------------------------------+---------------+
| `ProgressiveMultitaskRegressor`  | Classifier | Vector of      |                 | `CircularFingerprint`,                                         | `fit`         |
|                                  |            | shape `(N,)`   |                 | `RDKitDescriptors`, `CoulombMatrixEig`, `RdkitGridFeaturizer`, |               |
|                                  |            |                |                 | `BindingPocketFeaturizer`,                                     |               |
|                                  |            |                |                 | `AdjacencyFingerprint`, `ElementPropertyFingerprint`,          |               |
+----------------------------------+------------+----------------+-----------------+----------------------------------------------------------------+---------------+
| `RobustMultitaskClassifier`      | Classifier | Vector of      |                 | `CircularFingerprint`,                                         | `fit`         |
|                                  |            | shape `(N,)`   |                 | `RDKitDescriptors`, `CoulombMatrixEig`, `RdkitGridFeaturizer`, |               |
|                                  |            |                |                 | `BindingPocketFeaturizer`,                                     |               |
|                                  |            |                |                 | `AdjacencyFingerprint`, `ElementPropertyFingerprint`,          |               |
+----------------------------------+------------+----------------+-----------------+----------------------------------------------------------------+---------------+
| `RobustMultitaskRegressor`       | Classifier | Vector of      |                 | `CircularFingerprint`,                                         | `fit`         |
|                                  |            | shape `(N,)`   |                 | `RDKitDescriptors`, `CoulombMatrixEig`, `RdkitGridFeaturizer`, |               |
|                                  |            |                |                 | `BindingPocketFeaturizer`,                                     |               |
|                                  |            |                |                 | `AdjacencyFingerprint`, `ElementPropertyFingerprint`,          |               |
+----------------------------------+------------+----------------+-----------------+----------------------------------------------------------------+---------------+
| `ScScoreModel`                   | Classifier | Vector of      |                 | `CircularFingerprint`,                                         | `fit`         |
|                                  |            | shape `(N,)`   |                 | `RDKitDescriptors`, `CoulombMatrixEig`, `RdkitGridFeaturizer`, |               |
|                                  |            |                |                 | `BindingPocketFeaturizer`,                                     |               |
|                                  |            |                |                 | `AdjacencyFingerprint`, `ElementPropertyFingerprint`,          |               |
+----------------------------------+------------+----------------+-----------------+----------------------------------------------------------------+---------------+
| `SeqToSeq`                       | Sequence   | Sequence       |                 |                                                                |`fit_sequences`|
+----------------------------------+------------+----------------+-----------------+----------------------------------------------------------------+---------------+
| `Smiles2Vec`                     | Classifier/| Sequence       |                 | `SmilesToSeq`                                                  | `fit`         |
|                                  | Regressor  |                |                 |                                                                |               |
+----------------------------------+------------+----------------+-----------------+----------------------------------------------------------------+---------------+
| `TextCNNModel`                   | Classifier/| String         |                 |                                                                | `fit`         |
|                                  | Regressor  |                |                 |                                                                |               |
+----------------------------------+------------+----------------+-----------------+----------------------------------------------------------------+---------------+
| `WGAN`                           | Adversarial| Pair           |                 |                                                                |`fit_gan`      |
+----------------------------------+------------+----------------+-----------------+----------------------------------------------------------------+---------------+

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

.. _`Keras`: https://keras.io/


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

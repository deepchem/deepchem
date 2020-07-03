Model Classes
=============

DeepChem maintains an extensive collection of models for scientific applications.

Model Cheatsheet
----------------
If you're just getting started with DeepChem, you're probably interested in the
basics. The place to get started is this "model cheatsheet" that lists various
types of custom DeepChem models. Note that some wrappers like :code:`SklearnModel`
and :code:`XGBoostModel` which wrap external machine learning libraries are excluded,
but this table is otherwise complete.

As a note about how to read this table, each row describes what's needed to
invoke a given model. Some models must be applied with given :code:`Transformer` or
:code:`Featurizer` objects. Some models also have custom training methods. You can
read off what's needed to train the model from the table below.


+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| Model                                  | Type       | Input Type           | Transformations        | Acceptable Featurizers                                         | Fit Method           |
+========================================+============+======================+========================+================================================================+======================+
| :code:`AtomicConvModel`                | Classifier/| Tuple                |                        | :code:`ComplexNeighborListFragmentAtomicCoordinates`           | :code:`fit`          |
|                                        | Regressor  |                      |                        |                                                                |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`ChemCeption`                    | Classifier/| Tensor of shape      |                        | :code:`SmilesToImage`                                          | :code:`fit`          |
|                                        | Regressor  | :code:`(N, M, c)`    |                        |                                                                |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`CNN`                            | Classifier/| Tensor of shape      |                        |                                                                | :code:`fit`          |
|                                        | Regressor  | :code:`(N, c)` or    |                        |                                                                |                      |
|                                        |            | :code:`(N, M, c)` or |                        |                                                                |                      |
|                                        |            | :code:`(N, M, L, c)` |                        |                                                                |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`DTNNModel`                      | Classifier/| Matrix of            |                        | :code:`CoulombMatrix`                                          | :code:`fit`          |
|                                        | Regressor  | shape :code:`(N, N)` |                        |                                                                |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`DAGModel`                       | Classifier/| :code:`ConvMol`      | :code:`DAGTransformer` | :code:`ConvMolFeaturizer`                                      | :code:`fit`          |
|                                        | Regressor  |                      |                        |                                                                |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`GraphConvModel`                 | Classifier/| :code:`ConvMol`      |                        | :code:`ConvMolFeaturizer`                                      | :code:`fit`          |
|                                        | Regressor  |                      |                        |                                                                |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`MPNNModel`                      | Classifier/| :code:`WeaveMol`     |                        | :code:`WeaveFeaturizer`                                        | :code:`fit`          |
|                                        | Regressor  |                      |                        |                                                                |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`MultitaskClassifier`            | Classifier | Vector of            |                        | :code:`CircularFingerprint`,                                   | :code:`fit`          | 
|                                        |            | shape :code:`(N,)`   |                        | :code:`RDKitDescriptors`,                                      |                      |
|                                        |            |                      |                        | :code:`CoulombMatrixEig`,                                      |                      |
|                                        |            |                      |                        | :code:`RdkitGridFeaturizer`,                                   |                      |
|                                        |            |                      |                        | :code:`BindingPocketFeaturizer`,                               |                      |
|                                        |            |                      |                        | :code:`AdjacencyFingerprint`,                                  |                      |
|                                        |            |                      |                        | :code:`ElementPropertyFingerprint`,                            |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`MultitaskRegressor`             | Regressor  | Vector of            |                        | :code:`CircularFingerprint`,                                   | :code:`fit`          | 
|                                        |            | shape :code:`(N,)`   |                        | :code:`RDKitDescriptors`,                                      |                      |
|                                        |            |                      |                        | :code:`CoulombMatrixEig`,                                      |                      |
|                                        |            |                      |                        | :code:`RdkitGridFeaturizer`,                                   |                      |
|                                        |            |                      |                        | :code:`BindingPocketFeaturizer`,                               |                      |
|                                        |            |                      |                        | :code:`AdjacencyFingerprint`,                                  |                      |
|                                        |            |                      |                        | :code:`ElementPropertyFingerprint`,                            |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`MultitaskFitTransformRegressor` | Regressor  | Vector of            | Any                    | :code:`CircularFingerprint`,                                   | :code:`fit`          | 
|                                        |            | shape :code:`(N,)`   |                        | :code:`RDKitDescriptors`,                                      |                      |
|                                        |            |                      |                        | :code:`CoulombMatrixEig`,                                      |                      |
|                                        |            |                      |                        | :code:`RdkitGridFeaturizer`,                                   |                      |
|                                        |            |                      |                        | :code:`BindingPocketFeaturizer`,                               |                      |
|                                        |            |                      |                        | :code:`AdjacencyFingerprint`,                                  |                      |
|                                        |            |                      |                        | :code:`ElementPropertyFingerprint`,                            |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`MultitaskIRVClassifier`         | Classifier | Vector of            | :code:`IRVTransformer` | :code:`CircularFingerprint`,                                   | :code:`fit`          | 
|                                        |            | shape :code:`(N,)`   |                        | :code:`RDKitDescriptors`,                                      |                      |
|                                        |            |                      |                        | :code:`CoulombMatrixEig`,                                      |                      |
|                                        |            |                      |                        | :code:`RdkitGridFeaturizer`,                                   |                      |
|                                        |            |                      |                        | :code:`BindingPocketFeaturizer`,                               |                      |
|                                        |            |                      |                        | :code:`AdjacencyFingerprint`,                                  |                      |
|                                        |            |                      |                        | :code:`ElementPropertyFingerprint`,                            |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`ProgressiveMultitaskClassifier` | Classifier | Vector of            |                        | :code:`CircularFingerprint`,                                   | :code:`fit`          | 
|                                        |            | shape :code:`(N,)`   |                        | :code:`RDKitDescriptors`,                                      |                      |
|                                        |            |                      |                        | :code:`CoulombMatrixEig`,                                      |                      |
|                                        |            |                      |                        | :code:`RdkitGridFeaturizer`,                                   |                      |
|                                        |            |                      |                        | :code:`BindingPocketFeaturizer`,                               |                      |
|                                        |            |                      |                        | :code:`AdjacencyFingerprint`,                                  |                      |
|                                        |            |                      |                        | :code:`ElementPropertyFingerprint`,                            |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`ProgressiveMultitaskRegressor`  | Regressor  | Vector of            |                        | :code:`CircularFingerprint`,                                   | :code:`fit`          | 
|                                        |            | shape :code:`(N,)`   |                        | :code:`RDKitDescriptors`,                                      |                      |
|                                        |            |                      |                        | :code:`CoulombMatrixEig`,                                      |                      |
|                                        |            |                      |                        | :code:`RdkitGridFeaturizer`,                                   |                      |
|                                        |            |                      |                        | :code:`BindingPocketFeaturizer`,                               |                      |
|                                        |            |                      |                        | :code:`AdjacencyFingerprint`,                                  |                      |
|                                        |            |                      |                        | :code:`ElementPropertyFingerprint`,                            |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`RobustMultitaskClassifier`      | Classifier | Vector of            |                        | :code:`CircularFingerprint`,                                   | :code:`fit`          | 
|                                        |            | shape :code:`(N,)`   |                        | :code:`RDKitDescriptors`,                                      |                      |
|                                        |            |                      |                        | :code:`CoulombMatrixEig`,                                      |                      |
|                                        |            |                      |                        | :code:`RdkitGridFeaturizer`,                                   |                      |
|                                        |            |                      |                        | :code:`BindingPocketFeaturizer`,                               |                      |
|                                        |            |                      |                        | :code:`AdjacencyFingerprint`,                                  |                      |
|                                        |            |                      |                        | :code:`ElementPropertyFingerprint`,                            |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`RobustMultitaskRegressor`       | Regressor  | Vector of            |                        | :code:`CircularFingerprint`,                                   | :code:`fit`          | 
|                                        |            | shape :code:`(N,)`   |                        | :code:`RDKitDescriptors`,                                      |                      |
|                                        |            |                      |                        | :code:`CoulombMatrixEig`,                                      |                      |
|                                        |            |                      |                        | :code:`RdkitGridFeaturizer`,                                   |                      |
|                                        |            |                      |                        | :code:`BindingPocketFeaturizer`,                               |                      |
|                                        |            |                      |                        | :code:`AdjacencyFingerprint`,                                  |                      |
|                                        |            |                      |                        | :code:`ElementPropertyFingerprint`,                            |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`ScScoreModel`                   | Classifier | Vector of            |                        | :code:`CircularFingerprint`,                                   | :code:`fit`          | 
|                                        |            | shape :code:`(N,)`   |                        | :code:`RDKitDescriptors`,                                      |                      |
|                                        |            |                      |                        | :code:`CoulombMatrixEig`,                                      |                      |
|                                        |            |                      |                        | :code:`RdkitGridFeaturizer`,                                   |                      |
|                                        |            |                      |                        | :code:`BindingPocketFeaturizer`,                               |                      |
|                                        |            |                      |                        | :code:`AdjacencyFingerprint`,                                  |                      |
|                                        |            |                      |                        | :code:`ElementPropertyFingerprint`,                            |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`SeqToSeq`                       | Sequence   | Sequence             |                        |                                                                | :code:`fit_sequences`|
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`Smiles2Vec`                     | Classifier/| Sequence             |                        | :code:`SmilesToSeq`                                            | :code:`fit`          |
|                                        | Regressor  |                      |                        |                                                                |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`TextCNNModel`                   | Classifier/| String               |                        |                                                                | :code:`fit`          |
|                                        | Regressor  |                      |                        |                                                                |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`WGAN`                           | Adversarial| Pair                 |                        |                                                                | :code:`fit_gan`      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+

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

Model Cheatsheet
----------------
If you're just getting started with DeepChem, you're probably interested in the
basics. The place to get started is this "model cheatsheet" that lists various
types of custom DeepChem models. Note that some wrappers like :code:`SklearnModel`
and :code:`GBDTModel` which wrap external machine learning libraries are excluded,
but this table should otherwise be complete.

As a note about how to read these tables: Each row describes what's needed to
invoke a given model. Some models must be applied with given :code:`Transformer` or
:code:`Featurizer` objects. Most models can be trained calling :code:`model.fit`,
otherwise the name of the fit_method is given in the Comment column.
In order to run the models, make sure that the backend (Keras and tensorflow
or Pytorch or Jax) is installed.
You can thus read off what's needed to train the model from the table below.

**General purpose**

+--------------------------------+----------------------------------------------------------------+-------------+-----------------------------+---------+----------------------------+
| Model                          | Reference                                                      | Type        | Acceptable Featurizers      | Backend | Comment                    |
+================================+================================================================+=============+=============================+=========+============================+
| MultitaskClassifier            |                                                                | Classifier  | CircularFingerprint,        | PyTorch |                            |
|                                |                                                                |             | RDKitDescriptors,           |         |                            |
|                                |                                                                |             | CoulombMatrixEig,           |         |                            |
|                                |                                                                |             | RdkitGridFeaturizer,        |         |                            |
|                                |                                                                |             | BindingPocketFeaturizer,    |         |                            |
|                                |                                                                |             | ElementPropertyFingerprint  |         |                            |
+--------------------------------+----------------------------------------------------------------+-------------+-----------------------------+---------+----------------------------+
| MultitaskIRVClassifier         |                                                                | Classifier  | CircularFingerprint,        | Keras   | use :code:`IRVTransformer` |
|                                |                                                                |             | RDKitDescriptors,           |         |                            |
|                                |                                                                |             | CoulombMatrixEig,           |         |                            |
|                                |                                                                |             | RdkitGridFeaturizer,        |         |                            |
|                                |                                                                |             | BindingPocketFeaturizer,    |         |                            |
|                                |                                                                |             | ElementPropertyFingerprint  |         |                            |
+--------------------------------+----------------------------------------------------------------+-------------+-----------------------------+---------+----------------------------+
| ProgressiveMultitaskClassifier | `ref <https://arxiv.org/abs/1606.04671>`_                      | Classifier  | CircularFingerprint,        | Keras   |                            |
|                                |                                                                |             | RDKitDescriptors,           |         |                            |
|                                |                                                                |             | CoulombMatrixEig,           |         |                            |
|                                |                                                                |             | RdkitGridFeaturizer,        |         |                            |
|                                |                                                                |             | BindingPocketFeaturizer,    |         |                            |
|                                |                                                                |             | ElementPropertyFingerprint  |         |                            |
+--------------------------------+----------------------------------------------------------------+-------------+-----------------------------+---------+----------------------------+
| RobustMultitaskClassifier      | `ref <https://pubs.acs.org/doi/abs/10.1021/acs.jcim.7b00146>`_ | Classifier  | CircularFingerprint,        | Keras   |                            |
|                                |                                                                |             | RDKitDescriptors,           |         |                            |
|                                |                                                                |             | CoulombMatrixEig,           |         |                            |
|                                |                                                                |             | RdkitGridFeaturizer,        |         |                            |
|                                |                                                                |             | BindingPocketFeaturizer,    |         |                            |
|                                |                                                                |             | ElementPropertyFingerprint  |         |                            |
+--------------------------------+----------------------------------------------------------------+-------------+-----------------------------+---------+----------------------------+
| CNN                            |                                                                | Classifier/ |                             | Keras   |                            |
|                                |                                                                | Regressor   |                             |         |                            |
+--------------------------------+----------------------------------------------------------------+-------------+-----------------------------+---------+----------------------------+
| MultitaskFitTransformRegressor |                                                                | Regressor   | CircularFingerprint,        | PyTorch | any :code:`Transformer`    |
|                                |                                                                |             | RDKitDescriptors,           |         | can be used                |
|                                |                                                                |             | CoulombMatrixEig,           |         |                            |
|                                |                                                                |             | RdkitGridFeaturizer,        |         |                            |
|                                |                                                                |             | BindingPocketFeaturizer,    |         |                            |
|                                |                                                                |             | ElementPropertyFingerprint  |         |                            |
+--------------------------------+----------------------------------------------------------------+-------------+-----------------------------+---------+----------------------------+
| MultitaskRegressor             |                                                                | Regressor   | CircularFingerprint,        | PyTorch |                            |
|                                |                                                                |             | RDKitDescriptors,           |         |                            |
|                                |                                                                |             | CoulombMatrixEig,           |         |                            |
|                                |                                                                |             | RdkitGridFeaturizer,        |         |                            |
|                                |                                                                |             | BindingPocketFeaturizer,    |         |                            |
|                                |                                                                |             | ElementPropertyFingerprint, |         |                            |
+--------------------------------+----------------------------------------------------------------+-------------+-----------------------------+---------+----------------------------+
| ProgressiveMultitaskRegressor  | `ref <https://arxiv.org/abs/1606.04671>`_                      | Regressor   | CircularFingerprint,        | Keras   |                            |
|                                |                                                                |             | RDKitDescriptors,           |         |                            |
|                                |                                                                |             | CoulombMatrixEig,           |         |                            |
|                                |                                                                |             | RdkitGridFeaturizer,        |         |                            |
|                                |                                                                |             | BindingPocketFeaturizer,    |         |                            |
|                                |                                                                |             | ElementPropertyFingerprint  |         |                            |
+--------------------------------+----------------------------------------------------------------+-------------+-----------------------------+---------+----------------------------+
| RobustMultitaskRegressor       | `ref <https://pubs.acs.org/doi/abs/10.1021/acs.jcim.7b00146>`_ | Regressor   | CircularFingerprint,        | Keras   |                            |
|                                |                                                                |             | RDKitDescriptors,           |         |                            |
|                                |                                                                |             | CoulombMatrixEig,           |         |                            |
|                                |                                                                |             | RdkitGridFeaturizer,        |         |                            |
|                                |                                                                |             | BindingPocketFeaturizer,    |         |                            |
|                                |                                                                |             | ElementPropertyFingerprint  |         |                            |
+--------------------------------+----------------------------------------------------------------+-------------+-----------------------------+---------+----------------------------+
| WGAN                           | `ref <https://arxiv.org/abs/1701.07875>`_                      | Adversarial |                             | Keras   | fit method: fit_gan        |
+--------------------------------+----------------------------------------------------------------+-------------+-----------------------------+---------+----------------------------+
| SeqToSeq                       | `ref <https://arxiv.org/abs/1409.3215>`_                       |             |                             | Keras   | fit method: fit_sequences  |
+--------------------------------+----------------------------------------------------------------+-------------+-----------------------------+---------+----------------------------+

**Molecules**

Many models implemented in DeepChem were designed for small to medium-sized organic molecules,
most often drug-like compounds.
If your data is very different (e.g. molecules contain 'exotic' elements not present in the original dataset)
or cannot be represented well using SMILES (e.g. metal complexes, crystals), some adaptations to the
featurization and/or model might be needed to get reasonable results.

+------------------+----------------------------------------------------------------+-------------+----------------------------------------------+----------------------------+----------------------------+
| Model            | Reference                                                      | Type        | Acceptable Featurizers                       | Backend                    | Comment                    |
+==================+================================================================+=============+==============================================+============================+============================+
| ScScoreModel     | `ref <https://pubs.acs.org/doi/abs/10.1021/acs.jcim.7b00622>`_ | Classifier  | CircularFingerprint                          | Keras                      |                            |
+------------------+----------------------------------------------------------------+-------------+----------------------------------------------+----------------------------+----------------------------+
| AtomicConvModel  | `ref <https://arxiv.org/abs/1703.10603>`_                      | Classifier/ | ComplexNeighborListFragmentAtomicCoordinates | Keras                      |                            |
|                  |                                                                | Regressor   |                                              |                            |                            |
+------------------+----------------------------------------------------------------+-------------+----------------------------------------------+----------------------------+----------------------------+
| AttentiveFPModel | `ref <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ | Classifier/ | MolGraphConvFeaturizer                       | Torch                      |                            |
|                  |                                                                | Regressor   |                                              |                            |                            |
+------------------+----------------------------------------------------------------+-------------+----------------------------------------------+----------------------------+----------------------------+
| ChemCeption      | `ref <https://arxiv.org/abs/1706.06689>`_                      | Classifier/ | SmilesToImage                                | Keras                      |                            |
|                  |                                                                | Regressor   |                                              |                            |                            |
+------------------+----------------------------------------------------------------+-------------+----------------------------------------------+----------------------------+----------------------------+
| DAGModel         | `ref <https://pubs.acs.org/doi/abs/10.1021/ci400187y>`_        | Classifier/ | ConvMolFeaturizer                            | Keras                      | use :code:`DAGTransformer` |
|                  |                                                                | Regressor   |                                              |                            |                            |
+------------------+----------------------------------------------------------------+-------------+----------------------------------------------+----------------------------+----------------------------+
| GATModel         | `ref <https://arxiv.org/abs/1710.10903>`_                      | Classifier/ | MolGraphConvFeaturizer                       | DGL/PyTorch                |                            |
|                  |                                                                | Regressor   |                                              |                            |                            |
+------------------+----------------------------------------------------------------+-------------+----------------------------------------------+----------------------------+----------------------------+
| GCNModel         | `ref <https://arxiv.org/abs/1609.02907>`_                      | Classifier/ | MolGraphConvFeaturizer                       | DGL/PyTorch                |                            |
|                  |                                                                | Regressor   |                                              |                            |                            |
+------------------+----------------------------------------------------------------+-------------+----------------------------------------------+----------------------------+----------------------------+
| GraphConvModel   | `ref <https://arxiv.org/abs/1509.09292>`_                      | Classifier/ | ConvMolFeaturizer                            | Keras                      |                            |
|                  |                                                                | Regressor   |                                              |                            |                            |
+------------------+----------------------------------------------------------------+-------------+----------------------------------------------+----------------------------+----------------------------+
| MEGNetModel      | `ref <https://arxiv.org/abs/1812.05055>`_                      | Classifier/ |                                              | PyTorch, PyTorch Geometric |                            |
|                  |                                                                | Regressor   |                                              |                            |                            |
+------------------+----------------------------------------------------------------+-------------+----------------------------------------------+----------------------------+----------------------------+
| MPNNModel        | `ref <https://arxiv.org/abs/1704.01212>`_                      | Classifier/ | MolGraphConvFeaturizer                       | DGL/PyTorch                |                            |
|                  |                                                                | Regressor   |                                              |                            |                            |
+------------------+----------------------------------------------------------------+-------------+----------------------------------------------+----------------------------+----------------------------+
| PagtnModel       | `ref <https://arxiv.org/abs/1905.12712>`_                      | Classifier/ | PagtnMolGraphFeaturizer,                     | DGL/PyTorch                |                            |
|                  |                                                                | Regressor   | MolGraphConvFeaturizer                       |                            |                            |
+------------------+----------------------------------------------------------------+-------------+----------------------------------------------+----------------------------+----------------------------+
| Smiles2Vec       | `ref <https://arxiv.org/abs/1712.02034>`_                      | Classifier/ | SmilesToSeq                                  | Keras                      |                            |
|                  |                                                                | Regressor   |                                              |                            |                            |
+------------------+----------------------------------------------------------------+-------------+----------------------------------------------+----------------------------+----------------------------+
| TextCNNModel     | `ref <https://arxiv.org/abs/1705.10843>`_                      | Classifier/ |                                              | Keras                      |                            |
|                  |                                                                | Regressor   |                                              |                            |                            |
+------------------+----------------------------------------------------------------+-------------+----------------------------------------------+----------------------------+----------------------------+
| DTNNModel        | `ref <https://arxiv.org/abs/1609.08259>`_                      | Regressor   | CoulombMatrix                                | Keras                      |                            |
+------------------+----------------------------------------------------------------+-------------+----------------------------------------------+----------------------------+----------------------------+
| MATModel         | `ref <https://arxiv.org/abs/2002.08264>`_                      | Regressor   | MATFeaturizer                                | PyTorch                    |                            |
+------------------+----------------------------------------------------------------+-------------+----------------------------------------------+----------------------------+----------------------------+
| WeaveModel       | `ref <https://arxiv.org/abs/1603.00856>`_                      | Regressor   | WeaveFeaturizer                              | Keras                      |                            |
+------------------+----------------------------------------------------------------+-------------+----------------------------------------------+----------------------------+----------------------------+
| BasicMolGANModel | `ref <https://arxiv.org/abs/1805.11973>`_                      | Generator   | MolGanFeaturizer                             | Keras                      | fit method: fit_gan        |
+------------------+----------------------------------------------------------------+-------------+----------------------------------------------+----------------------------+----------------------------+

**Materials**

The following models were designed specifically for (inorganic) materials.

+-------------+------------------------------------------------------------+-------------+------------------------+-------------------+-------------------+
| Model       | Reference                                                  | Type        | Acceptable Featurizers | Backend           | Comment           |
+=============+============================================================+=============+========================+===================+===================+
| CGCNNModel  | `ref <https://arxiv.org/abs/1710.10324>`_                  | Classifier/ | CGCNNFeaturizer        | DGL/PyTorch       | crystal graph CNN |
|             |                                                            | Regressor   |                        |                   |                   |
+-------------+------------------------------------------------------------+-------------+------------------------+-------------------+-------------------+
| MEGNetModel | `ref <https://arxiv.org/abs/1812.05055>`_                  | Classifier/ |                        | PyTorch,          |                   |
|             |                                                            | Regressor   |                        | PyTorch Geometric |                   |
+-------------+------------------------------------------------------------+-------------+------------------------+-------------------+-------------------+
| LCNNModel   | `ref <https://pubs.acs.org/doi/10.1021/acs.jpcc.9b03370>`_ | Regressor   | LCNNFeaturizer         | PyTorch           | lattice CNN       |
+-------------+------------------------------------------------------------+-------------+------------------------+-------------------+-------------------+




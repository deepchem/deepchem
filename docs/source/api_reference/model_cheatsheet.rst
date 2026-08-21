Model Cheatsheet
======================

If you're just getting started with DeepChem, you're probably interested in the basics. The place to get started is this "model cheatsheet" that lists various types of custom DeepChem models. Note that some wrappers like `SklearnModel` and `GBDTModel`, which wrap external machine learning libraries, are excluded, but this table should otherwise be complete.

Each row describes what's needed to invoke a given model. Some models must be applied with given `Transformer` or `Featurizer` objects. Most models can be trained by calling `model.fit()`, otherwise, the required fit method is mentioned in the Comment column.

To run the models, ensure the appropriate backend (Keras/TensorFlow, PyTorch, or JAX) is installed. You can read off what's needed to train the model from the tables below.


General Purpose Models
======================


.. list-table:: General Purpose Models
   :widths: 25 20 20 20 30
   :header-rows: 1

   * - Model Name
     - Transformer Required
     - Featurizer Required
     - Fit Method
     - Comment
   * - TensorGraph
     - Optional
     - Optional
     - ``fit()``
     - Flexible model for custom architectures.
   * - KerasModel
     - Optional
     - Optional
     - ``fit()``
     - Wrapper for Keras models.
   * - PyTorchModel
     - Optional
     - Optional
     - ``fit()``
     - Wrapper for PyTorch models.
   * - JaxModel
     - Optional
     - Optional
     - ``fit()``
     - Wrapper for JAX models.


Molecular Models
================

Many models implemented in DeepChem were designed for small to medium-sized organic molecules, most often drug-like compounds. If your data is very different (e.g., molecules contain 'exotic' elements not present in the original dataset) or cannot be represented well using SMILES (e.g., metal complexes, crystals), some adaptations to the featurization and/or model might be needed to get reasonable results.


.. list-table:: Molecular Models
   :widths: 25 20 20 20 30
   :header-rows: 1

   * - Model Name
     - Transformer Required
     - Featurizer Required
     - Fit Method
     - Comment
   * - GraphConvModel
     - Yes
     - GraphConv
     - ``fit()``
     - Graph convolutional networks for molecules.
   * - WeaveModel
     - Yes
     - WeaveFeaturizer
     - ``fit()``
     - Weave networks for molecular graphs.
   * - MPNNModel
     - Yes
     - CoulombMatrix
     - ``fit()``
     - Message-passing neural networks.
   * - DTNNModel
     - Yes
     - DTNNFeaturizer
     - ``fit()``
     - Deep tensor neural networks.
   * - AttentiveFPModel
     - Yes
     - GraphConv
     - ``fit()``
     - Attention-based graph neural network.
   * - ANIModel
     - No
     - ANI Featurizer
     - ``fit()``
     - ANI deep learning potential for molecular dynamics.
   * - DMPNNModel
     - Yes
     - DMPNN Featurizer
     - ``fit()``
     - Directed message-passing neural network.


Material Models
===============

The following models were designed specifically for (inorganic) materials.


.. list-table:: Material Models
   :widths: 25 20 20 20 30
   :header-rows: 1

   * - Model Name
     - Transformer Required
     - Featurizer Required
     - Fit Method
     - Comment
   * - CGCNNModel
     - Yes
     - CGCNNFeaturizer
     - ``fit()``
     - Crystal Graph Convolutional Neural Networks.
   * - MEGNetModel
     - Yes
     - MEGNetFeaturizer
     - ``fit()``
     - MatErials Graph Networks.
   * - SchNetModel
     - Yes
     - SchNetFeaturizer
     - ``fit()``
     - SchNet for materials.
   * - LCNNModel
     - Yes
     - LatticeConvFeaturizer
     - ``fit()``
     - Lattice convolutional neural network.


Notes
=====

1. **Transformers and Featurizers**: Some models require specific transformers or featurizers to preprocess the data. Ensure these are applied before training.
2. **Fit Methods**: Most models use the ``fit()`` method for training. If a different method is required, it will be noted in the "Fit Method" column.
3. **Backend Requirements**: Ensure the appropriate backend (Keras/TensorFlow, PyTorch, or JAX) is installed for the model you intend to use.

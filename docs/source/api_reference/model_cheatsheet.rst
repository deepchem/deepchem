==================
Model Cheatsheet
==================

If you're just getting started with DeepChem, you're probably interested in the basics.  
This "model cheatsheet" provides an overview of various custom DeepChem models.  
Some wrappers, such as ``SklearnModel`` and ``GBDTModel``, which integrate external ML libraries, are excluded.  
Otherwise, this table provides a complete list of DeepChem models.

To use these models, ensure that the required backend (Keras/TensorFlow, PyTorch, or JAX) is installed.  
Each row in the tables below describes what's needed to invoke a given model, including required transformers, featurizers, and fit methods.



General Purpose Models
======================

These models are versatile and can be applied to a wide range of tasks.

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

Many DeepChem models are designed for small to medium-sized organic molecules, such as drug-like compounds.  
If your data includes "exotic" elements or cannot be represented well using SMILES (e.g., metal complexes, crystals), additional adaptations may be needed.

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

These models are designed specifically for (inorganic) materials, such as crystals and alloys.

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




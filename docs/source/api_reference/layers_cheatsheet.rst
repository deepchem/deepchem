Layers Cheatsheet
=================

This page provides a structured reference for commonly used layers in
DeepChem. It is intended as a quick comparison guide when building
custom neural network models.

Each entry includes:

- **Purpose** — What the layer is designed to do.
- **Key Arguments** — Frequently used constructor parameters.
- **Input / Output** — Expected tensor format.
- **Example** — Minimal usage snippet.

The layout mirrors ``models.rst`` for consistency across documentation.

.. contents::
   :local:
   :depth: 1


Core Neural Network Layers
==========================

Dense (Fully Connected)
-----------------------

**Purpose:** Applies a linear transformation to the input tensor.
Used in MLP blocks and output heads.

**Key Arguments:**
- ``in_channels``
- ``out_channels``
- ``activation`` (optional)

**Input / Output:**
- Input: ``(batch_size, in_channels)``
- Output: ``(batch_size, out_channels)``

**Example:**

.. code-block:: python

   from deepchem.models.layers import Dense

   layer = Dense(in_channels=128,
                 out_channels=64,
                 activation='relu')
   output = layer(x)


Dropout
-------

**Purpose:** Randomly zeroes input elements during training to
reduce overfitting.

**Key Arguments:**
- ``p`` — Probability of zeroing elements

**Input / Output:**
- Shape preserved.

**Example:**

.. code-block:: python

   from deepchem.models.layers import Dropout

   drop = Dropout(p=0.3)
   output = drop(x)


BatchNorm
---------

**Purpose:** Normalizes activations across a mini-batch to improve
training stability and convergence speed.

**Key Arguments:**
- ``num_features``

**Input / Output:**
- Shape preserved.

**Example:**

.. code-block:: python

   from deepchem.models.layers import BatchNorm

   bn = BatchNorm(num_features=64)
   output = bn(x)


Activation Layers
-----------------

**Purpose:** Introduce nonlinearity into the network.

Available examples:
- ``ReLU``
- ``LeakyReLU``
- ``Sigmoid``
- ``Tanh``

**Example:**

.. code-block:: python

   from deepchem.models.layers import ReLU

   act = ReLU()
   output = act(x)


Convolution Layers
==================

Conv1D
------

**Purpose:** 1D convolution for sequence or spectral data.

**Key Arguments:**
- ``in_channels``
- ``out_channels``
- ``kernel_size``
- ``stride``

**Example:**

.. code-block:: python

   from deepchem.models.layers import Conv1D

   conv = Conv1D(in_channels=32,
                 out_channels=64,
                 kernel_size=3,
                 stride=1)
   output = conv(sequence_tensor)


Conv2D
------

**Purpose:** 2D convolution for image-like inputs.

**Input Format:** ``(batch_size, channels, height, width)``

**Example:**

.. code-block:: python

   from deepchem.models.layers import Conv2D

   conv = Conv2D(in_channels=3,
                 out_channels=16,
                 kernel_size=3,
                 stride=1)
   output = conv(image_tensor)


Graph Neural Network Layers
===========================

GraphConv
---------

**Purpose:** Performs graph convolution via neighborhood message passing.
Commonly used in molecular property prediction.

**Key Arguments:**
- ``in_channels``
- ``out_channels``
- ``activation``

**Input:**
- ``node_features``
- ``edge_index`` (graph connectivity)

**Example:**

.. code-block:: python

   from deepchem.models.layers import GraphConv

   conv = GraphConv(in_channels=64,
                    out_channels=128,
                    activation='relu')
   node_rep = conv(node_features, edge_index)


GATConv
-------

**Purpose:** Graph Attention Convolution.
Learns attention weights for neighbor aggregation.

**Key Arguments:**
- ``in_channels``
- ``out_channels``
- ``num_heads``

**Example:**

.. code-block:: python

   from deepchem.models.layers import GATConv

   gat = GATConv(in_channels=64,
                 out_channels=64,
                 num_heads=4)
   output = gat(node_features, edge_index)


ResGatedGraphConv
-----------------

**Purpose:** Residual gated graph convolution.
Suitable for deeper graph architectures.

**Key Arguments:**
- ``in_channels``
- ``out_channels``
- ``activation``

**Example:**

.. code-block:: python

   from deepchem.models.layers import ResGatedGraphConv

   layer = ResGatedGraphConv(in_channels=64,
                             out_channels=64)
   output = layer(node_features, edge_index)


MessagePassing (Base Class)
---------------------------

**Purpose:** Base class for implementing custom message-passing layers.
Most GNN layers inherit from this class.

Users extending GNN functionality should subclass ``MessagePassing``.


Pooling Layers
==============

GlobalPool
----------

**Purpose:** Aggregates node features into a graph-level representation.

**Input:**
- Node feature tensor

**Output:**
- Graph-level feature vector

**Example:**

.. code-block:: python

   from deepchem.models.layers import GlobalPool

   pool = GlobalPool()
   graph_rep = pool(node_features)


AvgPool / MaxPool
-----------------

**Purpose:** Spatial or feature-wise pooling operations.

Used in CNN and GNN architectures.


Typical Layer Ordering
======================

For feed-forward blocks:

.. code-block:: python

   x = Dense(...)
   x = BatchNorm(...)
   x = ReLU(...)
   x = Dropout(...)

Recommended order:

1. Linear / Convolution
2. Normalization
3. Activation
4. Dropout


When to Use Which Layer
=======================

+------------------------+----------------------------------------+
| Layer Type             | Typical Use Case                      |
+========================+========================================+
| Dense                  | MLP blocks, prediction heads          |
+------------------------+----------------------------------------+
| GraphConv              | Molecular property prediction         |
+------------------------+----------------------------------------+
| GATConv                | Attention-based graph modeling        |
+------------------------+----------------------------------------+
| Conv1D                 | Sequence modeling                     |
+------------------------+----------------------------------------+
| Conv2D                 | Image-like data                       |
+------------------------+----------------------------------------+
| GlobalPool             | Graph-level embedding aggregation     |
+------------------------+----------------------------------------+


Extending This Cheatsheet
=========================

To add a new layer:

1. Locate the implementation under ``deepchem/models``.
2. Add a section following the existing structure.
3. Keep descriptions concise.
4. Provide a minimal working example.

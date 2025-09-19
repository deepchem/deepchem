=========================
DeepChem Layers Cheatsheet
=========================

DeepChem provides a variety of scientifically relevant differentiable layers.
This cheatsheet summarizes commonly used layers in DeepChem, along with their descriptions, example usage, and key parameters.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Layer Name
     - Description
     - Example Usage
   * - **GraphConv**
     - A graph convolution layer that applies graph convolutions to molecular graphs. Useful for processing molecular structures.
     - ``GraphConv(out_channels=128, activation_fn=torch.nn.ReLU(), dropout=0.2)``
   * - **WeaveLayer**
     - Weave model layer for learning atomic and pairwise features in molecular graphs.
     - ``WeaveLayer(n_atom_feat=75, n_pair_feat=14, init=deepchem.nn.initializations.xavier_normal)``
   * - **DAGLayer**
     - Directed Acyclic Graph layer for hierarchical message passing across molecular graphs.
     - ``DAGLayer(n_graph_feat=30, dropout=0.1, activation_fn=torch.nn.Sigmoid())``
   * - **GATLayer**
     - Graph Attention Network (GAT) layer that uses attention mechanisms for feature aggregation.
     - ``GATLayer(n_heads=8, out_channels=64, dropout=0.3, leaky_relu_slope=0.2)``
   * - **EdgeConv**
     - Edge-conditioned convolution for graph networks, allowing dynamic edge updates during training.
     - ``EdgeConv(in_channels=128, out_channels=64, batch_norm=True, activation_fn=torch.nn.LeakyReLU())``
   * - **MessagePassing**
     - A flexible message-passing framework for defining custom graph convolution operations.
     - ``MessagePassing(aggr='mean', flow='target_to_source', edge_dim=16)``

.. note::
   This list is non-exhaustive. Check the DeepChem documentation for additional layers, advanced configurations, and real-world use cases.

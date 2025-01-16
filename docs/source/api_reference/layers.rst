Layers
======
Deep learning models are often said to be made up of "layers".
Intuitively, a "layer" is a function which transforms some
tensor into another tensor. DeepChem maintains an extensive
collection of layers which perform various useful scientific
transformations. For now, most layers are Keras only but over
time we expect this support to expand to other types of models
and layers.

.. include:: layers_cheatsheet.rst

Keras Layers
------------

.. autoclass:: deepchem.models.layers.InteratomicL2Distances
  :members:

.. autoclass:: deepchem.models.layers.GraphConv
  :members:

.. autoclass:: deepchem.models.layers.GraphPool
  :members:

.. autoclass:: deepchem.models.layers.GraphGather
  :members:

.. autoclass:: deepchem.models.layers.MolGANConvolutionLayer
  :members:

.. autoclass:: deepchem.models.layers.MolGANAggregationLayer
  :members:

.. autoclass:: deepchem.models.layers.MolGANMultiConvolutionLayer
  :members:

.. autoclass:: deepchem.models.layers.MolGANEncoderLayer
  :members:

.. autoclass:: deepchem.models.layers.LSTMStep
  :members:

.. autoclass:: deepchem.models.layers.AttnLSTMEmbedding
  :members:

.. autoclass:: deepchem.models.layers.IterRefLSTMEmbedding
  :members:

.. autoclass:: deepchem.models.layers.SwitchedDropout
  :members:

.. autoclass:: deepchem.models.layers.WeightedLinearCombo
  :members:

.. autoclass:: deepchem.models.layers.CombineMeanStd
  :members:

.. autoclass:: deepchem.models.layers.Stack
  :members:

.. autoclass:: deepchem.models.layers.VinaFreeEnergy
  :members:

.. autoclass:: deepchem.models.layers.NeighborList
  :members:

.. autoclass:: deepchem.models.layers.AtomicConvolution
  :members:

.. autoclass:: deepchem.models.layers.AlphaShareLayer
  :members:

.. autoclass:: deepchem.models.layers.SluiceLoss
  :members:

.. autoclass:: deepchem.models.layers.BetaShare
  :members:

.. autoclass:: deepchem.models.layers.ANIFeat
  :members:

.. autoclass:: deepchem.models.layers.GraphEmbedPoolLayer
  :members:

.. autoclass:: deepchem.models.layers.GraphCNN
  :members:

.. autoclass:: deepchem.models.layers.Highway
  :members:

.. autoclass:: deepchem.models.layers.WeaveLayer
  :members:

.. autoclass:: deepchem.models.layers.WeaveGather
  :members:

.. autoclass:: deepchem.models.layers.DTNNEmbedding
  :members:

.. autoclass:: deepchem.models.layers.DTNNStep
  :members:

.. autoclass:: deepchem.models.layers.DTNNGather
  :members:

.. autoclass:: deepchem.models.layers.DAGLayer
  :members:

.. autoclass:: deepchem.models.layers.DAGGather
  :members:

.. autoclass:: deepchem.models.layers.MessagePassing
  :members:

.. autoclass:: deepchem.models.layers.EdgeNetwork
  :members:

.. autoclass:: deepchem.models.layers.GatedRecurrentUnit
  :members:

.. autoclass:: deepchem.models.layers.SetGather
  :members:

Torch Layers
------------

.. autoclass:: deepchem.models.torch_models.layers.AtomicConv
  :members:

.. autoclass:: deepchem.models.torch_models.layers.MultilayerPerceptron
  :members:

.. autoclass:: deepchem.models.torch_models.layers.CNNModule
  :members:

.. autoclass:: deepchem.models.torch_models.layers.ScaleNorm
  :members:

.. autoclass:: deepchem.models.torch_models.layers.MATEncoderLayer
  :members:

.. autoclass:: deepchem.models.torch_models.layers.MultiHeadedMATAttention
  :members:

.. autoclass:: deepchem.models.torch_models.layers.SublayerConnection
  :members:

.. autoclass:: deepchem.models.torch_models.layers.PositionwiseFeedForward
  :members:

.. autoclass:: deepchem.models.torch_models.layers.MATEmbedding
  :members:

.. autoclass:: deepchem.models.torch_models.layers.MATGenerator
  :members:

.. autofunction:: deepchem.models.layers.cosine_dist

.. autoclass:: deepchem.models.torch_models.layers.GraphNetwork
  :members:

.. autoclass:: deepchem.models.torch_models.layers.Affine
  :members:

.. autoclass:: deepchem.models.torch_models.layers.RealNVPLayer
  :members:

.. autoclass:: deepchem.models.torch_models.layers.DMPNNEncoderLayer
  :members:

.. autoclass:: deepchem.models.torch_models.InfoGraphEncoder
  :members:

.. autoclass:: deepchem.models.torch_models.GINEncoder
  :members:

.. autoclass:: deepchem.models.torch_models.layers.SetGather
  :members:

.. autoclass:: deepchem.models.torch_models.gnn.GNN
  :members:

.. autoclass:: deepchem.models.torch_models.gnn.GNNHead
  :members:

.. autoclass:: deepchem.models.torch_models.gnn.LocalGlobalDiscriminator
  :members:

.. autoclass:: deepchem.models.torch_models.pna_gnn.AtomEncoder
  :members:

.. autoclass:: deepchem.models.torch_models.pna_gnn.BondEncoder
  :members:

.. autoclass:: deepchem.models.torch_models.pna_gnn.PNALayer
  :members:

.. autoclass:: deepchem.models.torch_models.pna_gnn.PNAGNN
  :members:

.. autoclass:: deepchem.models.torch_models.PNA
  :members:

.. autoclass:: deepchem.models.torch_models.gnn3d.Net3DLayer
  :members:

.. autoclass:: deepchem.models.torch_models.gnn3d.Net3D
  :members:

.. autoclass:: deepchem.models.torch_models.layers.DTNNEmbedding
  :members:

.. autoclass:: deepchem.models.torch_models.layers.DTNNStep
  :members:

.. autoclass:: deepchem.models.torch_models.layers.DTNNGather
  :members:

.. autoclass:: deepchem.models.torch_models.gan.GradientPenaltyLayer
  :members:

.. autoclass:: deepchem.models.torch_models.layers.MolGANConvolutionLayer
  :members:

.. autoclass:: deepchem.models.torch_models.layers.MolGANAggregationLayer
  :members:


.. autoclass:: deepchem.models.torch_models.layers.MolGANMultiConvolutionLayer
  :members:

.. autoclass:: deepchem.models.torch_models.layers.MolGANEncoderLayer
  :members:

.. autoclass:: deepchem.models.torch_models.layers.EdgeNetwork
  :members:

.. autoclass:: deepchem.models.torch_models.layers.WeaveLayer
  :members:

.. autoclass:: deepchem.models.torch_models.layers.WeaveGather
  :members:


.. autoclass:: deepchem.models.torch_models.layers.MXMNetGlobalMessagePassing
  :members:

.. autoclass:: deepchem.models.torch_models.layers.MXMNetBesselBasisLayer
  :members:

.. autoclass:: deepchem.models.torch_models.dtnn.DTNN
  :members:

.. autoclass:: deepchem.models.torch_models.layers.VariationalRandomizer
  :members:

.. autoclass:: deepchem.models.torch_models.layers.EncoderRNN
  :members:

.. autoclass:: deepchem.models.torch_models.layers.DecoderRNN
  :members:

.. autoclass:: deepchem.models.torch_models.seqtoseq.SeqToSeq
  :members:

.. autoclass:: deepchem.models.torch_models.layers.FerminetElectronFeature
  :members:

.. autoclass:: deepchem.models.torch_models.layers.FerminetEnvelope
  :members:

.. autoclass:: deepchem.models.torch_models.layers.MXMNetLocalMessagePassing
  :members:

.. autoclass:: deepchem.models.torch_models.layers.MXMNetSphericalBasisLayer
  :members:

.. autoclass:: deepchem.models.torch_models.layers.HighwayLayer
  :members:

.. autoclass:: deepchem.models.torch_models.layers.GraphConv
  :members:

.. autoclass:: deepchem.models.torch_models.layers.GraphPool
  :members:

.. autoclass:: deepchem.models.torch_models.layers.GraphGather
  :members:

.. autoclass:: deepchem.models.torch_models.flows.ClampExp
  :members:

.. autoclass:: deepchem.models.torch_models.flows.ConstScaleLayer
  :members:

.. autofunction:: deepchem.models.torch_models.layers.cosine_dist

.. autoclass:: deepchem.models.torch_models.layers.DAGLayer
  :members:

.. autoclass:: deepchem.models.torch_models.layers.DAGGather
  :members:

Flow Layers
^^^^^^^^^^^

.. autoclass:: deepchem.models.torch_models.flows.Flow
  :members:

.. autoclass:: deepchem.models.torch_models.flows.Affine
  :members:

.. autoclass:: deepchem.models.torch_models.flows.MaskedAffineFlow
  :members:

.. autoclass:: deepchem.models.torch_models.flows.ActNorm
  :members:

.. autoclass:: deepchem.models.torch_models.flows.ClampExp
  :members:

.. autoclass:: deepchem.models.torch_models.flows.ConstScaleLayer
  :members:

.. autoclass:: deepchem.models.torch_models.flows.MLPFlow
  :members:

.. autoclass:: deepchem.models.torch_models.flows.NormalizingFlow
  :members:

Grover Layers
^^^^^^^^^^^^^

The following layers are used for implementing GROVER model as described in the paper `<Self-Supervised  Graph Transformer on Large-Scale Molecular Data <https://drug.ai.tencent.com/publications/GROVER.pdf>_`

.. autoclass:: deepchem.models.torch_models.grover_layers.GroverMPNEncoder
  :members:

.. autoclass:: deepchem.models.torch_models.grover_layers.GroverAttentionHead
  :members:

.. autoclass:: deepchem.models.torch_models.grover_layers.GroverMTBlock
  :members:

.. autoclass:: deepchem.models.torch_models.grover_layers.GroverTransEncoder
  :members:

.. autoclass:: deepchem.models.torch_models.grover_layers.GroverEmbedding
  :members:

.. autoclass:: deepchem.models.torch_models.grover_layers.GroverEmbedding
  :members:

.. autoclass:: deepchem.models.torch_models.grover_layers.GroverAtomVocabPredictor
  :members:

.. autoclass:: deepchem.models.torch_models.grover_layers.GroverBondVocabPredictor
  :members:

.. autoclass:: deepchem.models.torch_models.grover_layers.GroverFunctionalGroupPredictor
  :members:

.. autoclass:: deepchem.models.torch_models.grover.GroverPretrain
  :members:

.. autoclass:: deepchem.models.torch_models.grover.GroverFinetune
  :members:

.. autoclass:: deepchem.models.torch_models.grover.EquivariantLinear
  :members:

.. autoclass:: deepchem.models.torch_models.grover.SphericalHarmonics
  :members:
 
Attention Layers
^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.models.torch_models.attention.ScaledDotProductAttention
  :members:

.. autoclass:: deepchem.models.torch_models.attention.SelfAttention
  :members:

.. autoclass:: deepchem.models.torch_models.attention.SE3Attention
  :members:  

Readout Layers
^^^^^^^^^^^^^^

.. autoclass:: deepchem.models.torch_models.readout.GroverReadout
   :members:

Jax Layers
----------

.. autoclass:: deepchem.models.jax_models.layers.Linear
  :members:

Density Functional Theory Layers
--------------------------------

.. autoclass:: deepchem.models.dft.nnxc.BaseNNXC
   :members:

.. autoclass:: deepchem.models.dft.nnxc.NNLDA
   :members:

.. autoclass:: deepchem.models.dft.nnxc.NNPBE
   :members:

.. autoclass:: deepchem.models.dft.nnxc.HybridXC
   :members:

.. autoclass:: deepchem.models.dft.scf.XCNNSCF
   :members:

.. autoclass:: deepchem.models.dft.dftxc.DFTXC
   :members:

InceptionV3 Layers
------------------

.. autoclass:: deepchem.models.torch_models.inception_v3.InceptionA
   :members:

.. autoclass:: ddeepchem.models.torch_models.inception_v3.InceptionB
   :members:

.. autoclass:: deepchem.models.torch_models.inception_v3.InceptionC
   :members:

.. autoclass:: deepchem.models.torch_models.inception_v3.InceptionD
   :members:

.. autoclass:: deepchem.models.torch_models.inception_v3.InceptionE
   :members:

.. autoclass:: deepchem.models.torch_models.inception_v3.InceptionAux
   :members:

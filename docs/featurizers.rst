Featurizers
===========

DeepChem contains an extensive collection of featurizers. If you
haven't run into this terminology before, a "featurizer" is chunk of
code which transforms raw input data into a processed form suitable
for machine learning. Machine learning methods often need data to be
pre-chewed for them to process. Think of this like a mama penguin
chewing up food so the baby penguin can digest it easily.


Now if you've watched a few introductory deep learning lectures, you
might ask, why do we need something like a featurizer? Isn't part of
the promise of deep learning that we can learn patterns directly from
raw data?

Unfortunately it turns out that deep learning techniques need
featurizers just like normal machine learning methods do. Arguably,
they are less dependent on sophisticated featurizers and more capable
of learning sophisticated patterns from simpler data. But
nevertheless, deep learning systems can't simply chew up raw files.
For this reason, :code:`deepchem` provides an extensive collection of
featurization methods which we will review on this page.

Featurizer
----------

The :code:`dc.feat.Featurizer` class is the abstract parent class for all featurizers.

.. autoclass:: deepchem.feat.Featurizer
  :members:

MolecularFeaturizer
-------------------

Molecular Featurizers are those that work with datasets of molecules.

.. autoclass:: deepchem.feat.MolecularFeaturizer
  :members:

Here are some constants that are used by the graph convolutional featurizers for molecules.

.. autoclass:: deepchem.feat.graph_features.GraphConvConstants
  :members:
  :undoc-members:

There are a number of helper methods used by the graph convolutional classes which we document here.

.. autofunction:: deepchem.feat.graph_features.one_of_k_encoding

.. autofunction:: deepchem.feat.graph_features.one_of_k_encoding_unk

.. autofunction:: deepchem.feat.graph_features.get_intervals

.. autofunction:: deepchem.feat.graph_features.safe_index

.. autofunction:: deepchem.feat.graph_features.get_feature_list

.. autofunction:: deepchem.feat.graph_features.features_to_id

.. autofunction:: deepchem.feat.graph_features.id_to_features

.. autofunction:: deepchem.feat.graph_features.atom_to_id

This function helps compute distances between atoms from a given base atom.

.. autofunction:: deepchem.feat.graph_features.find_distance

This function is important and computes per-atom feature vectors used by
graph convolutional featurizers. 

.. autofunction:: deepchem.feat.graph_features.atom_features

This function computes the bond features used by graph convolutional
featurizers.

.. autofunction:: deepchem.feat.graph_features.bond_features

This function computes atom-atom features (for atom pairs which may not have bonds between them.)

.. autofunction:: deepchem.feat.graph_features.pair_features

ConvMolFeaturizer
^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.ConvMolFeaturizer
  :members:

WeaveFeaturizer
^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.WeaveFeaturizer
  :members:

CircularFingerprint
^^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.CircularFingerprint
  :members:

Mol2VecFingerprint
^^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.Mol2VecFingerprint
  :members:

RDKitDescriptors
^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.RDKitDescriptors
  :members:

MordredDescriptors
^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.MordredDescriptors
  :members:

CoulombMatrix
^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.CoulombMatrix
  :members:

CoulombMatrixEig
^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.CoulombMatrixEig
  :members:

AtomCoordinates
^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.AtomicCoordinates
  :members:

SmilesToSeq
^^^^^^^^^^^

.. autoclass:: deepchem.feat.SmilesToSeq
  :members:

SmilesToImage
^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.SmilesToImage
  :members:

OneHotFeaturizer
^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.OneHotFeaturizer
  :members:

ComplexFeaturizer
-----------------

The :code:`dc.feat.ComplexFeaturizer` class is the abstract parent class for all featurizers that work with three dimensional molecular complexes. 


.. autoclass:: deepchem.feat.ComplexFeaturizer
  :members:

RdkitGridFeaturizer
^^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.RdkitGridFeaturizer
  :members:

AtomConvFeaturizer
^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.NeighborListComplexAtomicCoordinates
  :members:

MaterialStructureFeaturizer
---------------------------

Material Structure Featurizers are those that work with datasets of crystals with
periodic boundary conditions. For inorganic crystal structures, these
featurizers operate on pymatgen.Structure objects, which include a
lattice and 3D coordinates that specify a periodic crystal structure. 
They should be applied on systems that have periodic boundary conditions.
Structure featurizers are not designed to work with molecules. 

.. autoclass:: deepchem.feat.MaterialStructureFeaturizer
  :members:

SineCoulombMatrix
^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.SineCoulombMatrix
  :members:

CGCNNFeaturizer
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.CGCNNFeaturizer
  :members:

MaterialCompositionFeaturizer
-----------------------------

Material Composition Featurizers are those that work with datasets of crystal
compositions with periodic boundary conditions. 
For inorganic crystal structures, these featurizers operate on chemical
compositions (e.g. "MoS2"). They should be applied on systems that have
periodic boundary conditions. Composition featurizers are not designed 
to work with molecules. 

.. autoclass:: deepchem.feat.MaterialCompositionFeaturizer
  :members:

ElementPropertyFingerprint
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.ElementPropertyFingerprint
  :members:

BindingPocketFeaturizer
-----------------------

.. autoclass:: deepchem.feat.BindingPocketFeaturizer
  :members:

UserDefinedFeaturizer
---------------------

.. autoclass:: deepchem.feat.UserDefinedFeaturizer
  :members:

BPSymmetryFunctionInput
-----------------------

.. autoclass:: deepchem.feat.BPSymmetryFunctionInput
  :members:

RawFeaturizer
-------------

.. autoclass:: deepchem.feat.RawFeaturizer
  :members:

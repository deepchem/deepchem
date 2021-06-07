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

.. contents:: Contents
    :local:


Molecule Featurizers
---------------------

These featurizers work with datasets of molecules.

Graph Convolution Featurizers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We are simplifying our graph convolution models by a joint data representation (:code:`GraphData`)
in a future version of DeepChem, so we provide several featurizers.

:code:`ConvMolFeaturizer` and :code:`WeaveFeaturizer` are used
with graph convolution models  which inherited :code:`KerasModel`.
:code:`ConvMolFeaturizer` is used with graph convolution models
except :code:`WeaveModel`. :code:`WeaveFeaturizer` are only used with :code:`WeaveModel`.
On the other hand, :code:`MolGraphConvFeaturizer` is used
with graph convolution models which inherited :code:`TorchModel`.
:code:`MolGanFeaturizer` will be used with MolGAN model,
a GAN model for generation of small molecules.

ConvMolFeaturizer
*****************

.. autoclass:: deepchem.feat.ConvMolFeaturizer
  :members:
  :inherited-members:

WeaveFeaturizer
***************

.. autoclass:: deepchem.feat.WeaveFeaturizer
  :members:
  :inherited-members:

MolGanFeaturizer
**********************

.. autoclass:: deepchem.feat.MolGanFeaturizer
  :members:
  :inherited-members:

MolGraphConvFeaturizer
**********************

.. autoclass:: deepchem.feat.MolGraphConvFeaturizer
  :members:
  :inherited-members:

PagtnMolGraphFeaturizer
**********************

.. autoclass:: deepchem.feat.PagtnMolGraphFeaturizer
  :members:
  :inherited-members:

Utilities
*********

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


MACCSKeysFingerprint
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.MACCSKeysFingerprint
  :members:

CircularFingerprint
^^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.CircularFingerprint
  :members:
  :inherited-members:

PubChemFingerprint
^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.PubChemFingerprint
  :members:

Mol2VecFingerprint
^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.Mol2VecFingerprint
  :members:
  :inherited-members:

RDKitDescriptors
^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.RDKitDescriptors
  :members:
  :inherited-members:

MordredDescriptors
^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.MordredDescriptors
  :members:
  :inherited-members:

CoulombMatrix
^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.CoulombMatrix
  :members:
  :inherited-members:

CoulombMatrixEig
^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.CoulombMatrixEig
  :members:
  :inherited-members:

AtomCoordinates
^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.AtomicCoordinates
  :members:
  :inherited-members:

BPSymmetryFunctionInput
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.BPSymmetryFunctionInput
  :members:
  :inherited-members:

SmilesToSeq
^^^^^^^^^^^

.. autoclass:: deepchem.feat.SmilesToSeq
  :members:
  :inherited-members:

SmilesToImage
^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.SmilesToImage
  :members:
  :inherited-members:

OneHotFeaturizer
^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.OneHotFeaturizer
  :members:
  :inherited-members:

RawFeaturizer
^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.RawFeaturizer
  :members:
  :inherited-members:


Molecular Complex Featurizers
-------------------------------

These featurizers work with three dimensional molecular complexes.

RdkitGridFeaturizer
^^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.RdkitGridFeaturizer
  :members:
  :inherited-members:

AtomicConvFeaturizer
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.AtomicConvFeaturizer
  :members:
  :inherited-members:


Inorganic Crystal Featurizers
------------------------------

These featurizers work with datasets of inorganic crystals.

MaterialCompositionFeaturizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Material Composition Featurizers are those that work with datasets of crystal
compositions with periodic boundary conditions. 
For inorganic crystal structures, these featurizers operate on chemical
compositions (e.g. "MoS2"). They should be applied on systems that have
periodic boundary conditions. Composition featurizers are not designed 
to work with molecules. 

ElementPropertyFingerprint
**************************

.. autoclass:: deepchem.feat.ElementPropertyFingerprint
  :members:
  :inherited-members:

ElemNetFeaturizer
*****************

.. autoclass:: deepchem.feat.ElemNetFeaturizer
  :members:

MaterialStructureFeaturizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Material Structure Featurizers are those that work with datasets of crystals with
periodic boundary conditions. For inorganic crystal structures, these
featurizers operate on pymatgen.Structure objects, which include a
lattice and 3D coordinates that specify a periodic crystal structure. 
They should be applied on systems that have periodic boundary conditions.
Structure featurizers are not designed to work with molecules. 

SineCoulombMatrix
*****************

.. autoclass:: deepchem.feat.SineCoulombMatrix
  :members:
  :inherited-members:

CGCNNFeaturizer
***************

.. autoclass:: deepchem.feat.CGCNNFeaturizer
  :members:
  :inherited-members:

LCNNFeaturizer
^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.LCNNFeaturizer
  :members:
  :inherited-members:


MaterialCompositionFeaturizer
-----------------------------

Molecule Tokenizers
-------------------

A tokenizer is in charge of preparing the inputs for a natural language processing model. 
For many scientific applications, it is possible to treat inputs as "words"/"sentences" and
use NLP methods to make meaningful predictions. For example, SMILES strings or DNA sequences
have grammatical structure and can be usefully modeled with NLP techniques. DeepChem provides
some scientifically relevant tokenizers for use in different applications. These tokenizers are
based on those from the Huggingface transformers library (which DeepChem tokenizers inherit from).

The base classes PreTrainedTokenizer and PreTrainedTokenizerFast implements the common methods
for encoding string inputs in model inputs and instantiating/saving python tokenizers
either from a local file or directory or from a pretrained tokenizer provided by the library
(downloaded from HuggingFace’s AWS S3 repository).

PreTrainedTokenizer `(transformers.PreTrainedTokenizer)`_ thus implements
the main methods for using all the tokenizers:

- Tokenizing (spliting strings in sub-word token strings), converting tokens strings to ids and back, and encoding/decoding (i.e. tokenizing + convert to integers)
- Adding new tokens to the vocabulary in a way that is independent of the underlying structure (BPE, SentencePiece…)
- Managing special tokens like mask, beginning-of-sentence, etc tokens (adding them, assigning them to attributes in the tokenizer for easy access and making sure they are not split during tokenization)

BatchEncoding holds the output of the tokenizer’s encoding methods
(__call__, encode_plus and batch_encode_plus) and is derived from a Python dictionary.
When the tokenizer is a pure python tokenizer, this class behave just like a standard python dictionary
and hold the various model inputs computed by these methodes (input_ids, attention_mask…).
For more details on the base tokenizers which the DeepChem tokenizers inherit from,
please refer to the following: `HuggingFace tokenizers docs`_

Tokenization methods on string-based corpuses in the life sciences are 
becoming increasingly popular for NLP-based applications to chemistry and biology.
One such example is ChemBERTa, a transformer for molecular property prediction.
DeepChem offers a tutorial for utilizing ChemBERTa using an alternate tokenizer,
a Byte-Piece Encoder, which can be found `here.`_

.. _`(transformers.PreTrainedTokenizer)`: https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer
.. _`HuggingFace tokenizers docs`: https://huggingface.co/transformers/main_classes/tokenizer.html
.. _`here.`: https://github.com/deepchem/deepchem/blob/master/examples/tutorials/22_Transfer_Learning_With_HuggingFace_tox21.ipynb

SmilesTokenizer
^^^^^^^^^^^^^^^

The :code:`dc.feat.SmilesTokenizer` module inherits from the BertTokenizer class in transformers.
It runs a WordPiece tokenization algorithm over SMILES strings using the tokenisation SMILES regex developed by Schwaller et. al.

The SmilesTokenizer employs an atom-wise tokenization strategy using the following Regex expression: ::

    SMI_REGEX_PATTERN = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#||\+|\\\\\/|:||@|\?|>|\*|\$|\%[0–9]{2}|[0–9])"


To use, please install the transformers package using the following pip command: ::

    pip install transformers

References:

-  `RXN Mapper: Unsupervised Attention-Guided Atom-Mapping`_
-  `Molecular Transformer: Unsupervised Attention-Guided Atom-Mapping`_

.. autoclass:: deepchem.feat.SmilesTokenizer
  :members:

BasicSmilesTokenizer
^^^^^^^^^^^^^^^^^^^^

The :code:`dc.feat.BasicSmilesTokenizer` module uses a regex tokenization pattern to tokenise SMILES strings.
The regex is developed by Schwaller et. al. The tokenizer is to be used on SMILES in cases
where the user wishes to not rely on the transformers API.

References:

-  `Molecular Transformer: Unsupervised Attention-Guided Atom-Mapping`_

.. autoclass:: deepchem.feat.BasicSmilesTokenizer
  :members:

.. _`RXN Mapper: Unsupervised Attention-Guided Atom-Mapping`: https://chemrxiv.org/articles/Unsupervised_Attention-Guided_Atom-Mapping/12298559
.. _`Molecular Transformer: Unsupervised Attention-Guided Atom-Mapping`: https://pubs.acs.org/doi/10.1021/acscentsci.9b00576

Other Featurizers
-----------------

BindingPocketFeaturizer
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.BindingPocketFeaturizer
  :members:
  :inherited-members:

UserDefinedFeaturizer
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: deepchem.feat.UserDefinedFeaturizer
  :members:
  :inherited-members:

Base Featurizers (for develop)
------------------------------

Featurizer
^^^^^^^^^^

The :code:`dc.feat.Featurizer` class is the abstract parent class for all featurizers.

.. autoclass:: deepchem.feat.Featurizer
  :members:

MolecularFeaturizer
^^^^^^^^^^^^^^^^^^^

If you're creating a new featurizer that featurizes molecules,
you will want to inherit from the abstract :code:`MolecularFeaturizer` base class.
This featurizer can take RDKit mol objects or SMILES as inputs.

.. autoclass:: deepchem.feat.MolecularFeaturizer
  :members:

MaterialCompositionFeaturizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you're creating a new featurizer that featurizes compositional formulas,
you will want to inherit from the abstract :code:`MaterialCompositionFeaturizer` base class.

.. autoclass:: deepchem.feat.MaterialCompositionFeaturizer
  :members:

MaterialStructureFeaturizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you're creating a new featurizer that featurizes inorganic crystal structure,
you will want to inherit from the abstract :code:`MaterialCompositionFeaturizer` base class.
This featurizer can take pymatgen structure objects or dictionaries as inputs.

.. autoclass:: deepchem.feat.MaterialStructureFeaturizer
  :members:

ComplexFeaturizer
^^^^^^^^^^^^^^^^^

If you're creating a new featurizer that featurizes a pair of ligand molecules and proteins,
you will want to inherit from the abstract :code:`ComplexFeaturizer` base class.
This featurizer can take a pair of PDB or SDF files which contain ligand molecules and proteins.

.. autoclass:: deepchem.feat.ComplexFeaturizer
  :members:

"""
Making it easy to import in classes.
"""
# flake8: noqa

# base classes for featurizers
from deepchem.feat.base_classes import Featurizer
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.feat.base_classes import MaterialStructureFeaturizer
from deepchem.feat.base_classes import MaterialCompositionFeaturizer
from deepchem.feat.base_classes import ComplexFeaturizer
from deepchem.feat.base_classes import UserDefinedFeaturizer

from deepchem.feat.graph_features import ConvMolFeaturizer
from deepchem.feat.graph_features import WeaveFeaturizer
from deepchem.feat.rdkit_grid_featurizer import RdkitGridFeaturizer
from deepchem.feat.binding_pocket_features import BindingPocketFeaturizer
from deepchem.feat.atomic_coordinates import AtomicCoordinates
from deepchem.feat.atomic_coordinates import NeighborListComplexAtomicCoordinates

# molecule featurizers
from deepchem.feat.molecule_featurizers import MolGraphConvFeaturizer
from deepchem.feat.molecule_featurizers import CircularFingerprint
from deepchem.feat.molecule_featurizers import CoulombMatrix
from deepchem.feat.molecule_featurizers import CoulombMatrixEig
from deepchem.feat.molecule_featurizers import MordredDescriptors
from deepchem.feat.molecule_featurizers import Mol2VecFingerprint
from deepchem.feat.molecule_featurizers import OneHotFeaturizer
from deepchem.feat.molecule_featurizers import RawFeaturizer
from deepchem.feat.molecule_featurizers import RDKitDescriptors
from deepchem.feat.molecule_featurizers import SmilesToImage
from deepchem.feat.molecule_featurizers import SmilesToSeq, create_char_to_idx

# material featurizers
from deepchem.feat.material_featurizers import ElementPropertyFingerprint
from deepchem.feat.material_featurizers import SineCoulombMatrix
from deepchem.feat.material_featurizers import CGCNNFeaturizer

try:
  import transformers
  from transformers import BertTokenizer

  from deepchem.feat.smiles_tokenizer import SmilesTokenizer
  from deepchem.feat.smiles_tokenizer import BasicSmilesTokenizer
except ModuleNotFoundError:
  pass

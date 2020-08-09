"""
Making it easy to import in classes.
"""
# flake8: noqa
from deepchem.feat.base_classes import Featurizer
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.feat.base_classes import MaterialStructureFeaturizer
from deepchem.feat.base_classes import MaterialCompositionFeaturizer
from deepchem.feat.base_classes import ComplexFeaturizer
from deepchem.feat.base_classes import UserDefinedFeaturizer

from deepchem.feat.graph_features import ConvMolFeaturizer
from deepchem.feat.graph_features import WeaveFeaturizer
from deepchem.feat.coulomb_matrices import BPSymmetryFunctionInput
from deepchem.feat.rdkit_grid_featurizer import RdkitGridFeaturizer
from deepchem.feat.binding_pocket_features import BindingPocketFeaturizer
from deepchem.feat.raw_featurizer import RawFeaturizer
from deepchem.feat.atomic_coordinates import AtomicCoordinates
from deepchem.feat.atomic_coordinates import NeighborListComplexAtomicCoordinates
from deepchem.feat.smiles_featurizers import SmilesToSeq, SmilesToImage

from deepchem.feat.molecule_featurizers import AdjacencyFingerprint
from deepchem.feat.molecule_featurizers import CircularFingerprint
from deepchem.feat.molecule_featurizers import CoulombMatrix
from deepchem.feat.molecule_featurizers import CoulombMatrixEig
from deepchem.feat.molecule_featurizers import OneHotFeaturizer
from deepchem.feat.molecule_featurizers import RDKitDescriptors

from deepchem.feat.material_featurizers import ElementPropertyFingerprint
from deepchem.feat.material_featurizers import SineCoulombMatrix
from deepchem.feat.material_featurizers import CGCNNFeaturizer

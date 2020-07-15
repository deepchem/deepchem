"""
Making it easy to import in classes.
"""
from deepchem.feat.base_featurizers import Featurizer
from deepchem.feat.base_featurizers import MolecularFeaturizer
from deepchem.feat.base_featurizers import ComplexFeaturizer
from deepchem.feat.base_featurizers import UserDefinedFeaturizer
from deepchem.feat.graph_features import ConvMolFeaturizer
from deepchem.feat.graph_features import WeaveFeaturizer
from deepchem.feat.fingerprints import CircularFingerprint
from deepchem.feat.coulomb_matrices import CoulombMatrix
from deepchem.feat.coulomb_matrices import CoulombMatrixEig
from deepchem.feat.coulomb_matrices import BPSymmetryFunctionInput
from deepchem.feat.rdkit_descriptors import RDKitDescriptors
from deepchem.feat.rdkit_grid_featurizer import RdkitGridFeaturizer
from deepchem.feat.binding_pocket_features import BindingPocketFeaturizer
from deepchem.feat.one_hot import OneHotFeaturizer
from deepchem.feat.raw_featurizer import RawFeaturizer
from deepchem.feat.atomic_coordinates import AtomicCoordinates
from deepchem.feat.atomic_coordinates import NeighborListComplexAtomicCoordinates
from deepchem.feat.adjacency_fingerprints import AdjacencyFingerprint
from deepchem.feat.smiles_featurizers import SmilesToSeq, SmilesToImage
from deepchem.feat.materials_featurizers import ElementPropertyFingerprint, SineCoulombMatrix, StructureGraphFeaturizer

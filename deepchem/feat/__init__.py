"""
Making it easy to import in classes.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

from deepchem.feat.base_classes import Featurizer
from deepchem.feat.base_classes import ComplexFeaturizer
from deepchem.feat.base_classes import UserDefinedFeaturizer
from deepchem.feat.graph_features import ConvMolFeaturizer
from deepchem.feat.graph_features import WeaveFeaturizer
from deepchem.feat.fingerprints import CircularFingerprint
from deepchem.feat.basic import RDKitDescriptors
from deepchem.feat.coulomb_matrices import CoulombMatrix
from deepchem.feat.coulomb_matrices import CoulombMatrixEig
from deepchem.feat.coulomb_matrices import BPSymmetryFunction
from deepchem.feat.rdkit_grid_featurizer import RdkitGridFeaturizer
from deepchem.feat.nnscore_utils import hydrogenate_and_compute_partial_charges
from deepchem.feat.binding_pocket_features import BindingPocketFeaturizer
from deepchem.feat.one_hot import OneHotFeaturizer
from deepchem.feat.raw_featurizer import RawFeaturizer
from deepchem.feat.atomic_coordinates import AtomicCoordinates
from deepchem.feat.atomic_coordinates import NeighborListComplexAtomicCoordinates
from deepchem.feat.adjacency_fingerprints import AdjacencyFingerprint
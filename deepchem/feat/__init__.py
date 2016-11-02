"""
Making it easy to import in classes.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

# TODO(rbharath): Handle this * import and replace with explicit imports later
from deepchem.feat.base_classes import *
from deepchem.feat.featurize import DataLoader
from deepchem.feat.graph_features import ConvMolFeaturizer
from deepchem.feat.fingerprints import CircularFingerprint
from deepchem.feat.basic import RDKitDescriptors
from deepchem.feat.coulomb_matrices import CoulombMatrixEig

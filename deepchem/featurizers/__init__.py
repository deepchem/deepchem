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
from deepchem.featurizers.base_classes import *
from deepchem.featurizers.featurize import DataLoader
from deepchem.featurizers.graph_features import ConvMolFeaturizer
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.featurizers.basic import RDKitDescriptors

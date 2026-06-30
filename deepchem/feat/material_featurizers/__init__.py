"""
Featurizers for inorganic crystals.
"""
# flake8: noqa
import logging

logger = logging.getLogger(__name__)

from deepchem.feat.material_featurizers.element_property_fingerprint import ElementPropertyFingerprint
from deepchem.feat.material_featurizers.sine_coulomb_matrix import SineCoulombMatrix
from deepchem.feat.material_featurizers.cgcnn_featurizer import CGCNNFeaturizer
from deepchem.feat.material_featurizers.elemnet_featurizer import ElemNetFeaturizer
from deepchem.feat.material_featurizers.lcnn_featurizer import LCNNFeaturizer
try:
    from deepchem.feat.material_featurizers.atomistic_radius_graph_featurizer import AtomisticRadiusGraphFeaturizer
except ModuleNotFoundError as e:
    logger.warning(
        f'Skipped loading material featurizer, missing a dependency. {e}')

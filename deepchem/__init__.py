"""
Imports all submodules
"""
import os

# TensorFlow >=2.16 defaults to Keras 3, but deepchem's TF-based models rely on
# the Keras 2 API (e.g. tf.keras.optimizers.legacy). This must be set before
# tensorflow is imported anywhere, including by submodules below.
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

# If you push the tag, please remove `.dev`
__version__ = '2.8.1.dev'

import deepchem.data
import deepchem.feat
import deepchem.hyper
import deepchem.metalearning
import deepchem.metrics
import deepchem.models
import deepchem.splits
import deepchem.trans
import deepchem.utils
import deepchem.dock
import deepchem.molnet
import deepchem.rl

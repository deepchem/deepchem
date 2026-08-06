"""
Imports all submodules
"""
import os

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

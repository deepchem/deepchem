"""
Imports all submodules
"""
from __future__ import division
from __future__ import unicode_literals
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

__version__ = '2.2.0'

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

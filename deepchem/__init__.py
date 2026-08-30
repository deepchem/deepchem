"""
Imports all submodules
"""

# If you push the tag, please remove `.dev`
__version__ = '2.8.1.dev'

from . import data as data
from . import feat as feat
from . import hyper as hyper
from . import metalearning as metalearning
from . import metrics as metrics
from . import models as models
from . import splits as splits
from . import trans as trans
from . import utils as utils
from . import dock as dock
from . import molnet as molnet
from . import rl as rl

__all__ = [
    "data",
    "feat",
    "hyper",
    "metalearning",
    "metrics",
    "models",
    "splits",
    "trans",
    "utils",
    "dock",
    "molnet",
    "rl",
]

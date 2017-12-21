"""
Contains Sequential model adapted from keras/keras/models.py.

This class is adapted from Keras directly. Have cut out functionality
and changed API to match DeepChem style.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2017, Stanford University"
__license__ = "MIT"

import time
import os
import tempfile
import numpy as np
import tensorflow as tf
from deepchem.models.models import Model

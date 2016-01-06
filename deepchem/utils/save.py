"""
Utility functions to save keras/sklearn models.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
from deepchem.models import get_model_type 
from deepchem.models import get_model_filename
from keras.models import model_from_json
from sklearn.externals import joblib


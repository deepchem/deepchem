"""
Code for processing the Google vs-datasets.
"""
__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2015, Stanford University"
__license__ = "LGPL"

import os
import numpy as np
import gzip
import cPickle as pickle
import csv
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

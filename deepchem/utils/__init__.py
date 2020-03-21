"""
Miscellaneous utility functions.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "BSD 3-clause"

import gzip
import numpy as np
import os
import pandas as pd
import sys
import tempfile
import tarfile
import zipfile

try:
  from urllib.request import urlretrieve  # Python 3
except:
  from urllib import urlretrieve  # Python 2



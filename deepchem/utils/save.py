"""
Simple utils to save and load from disk.
"""
import joblib
import gzip
import json
import pickle
import pandas as pd
import numpy as np
import os
import deepchem
import warnings
from deepchem.utils.genomics import encode_bio_sequence as encode_sequence, encode_fasta_sequence as fasta_sequence, seq_one_hot_encode as seq_one_hotencode



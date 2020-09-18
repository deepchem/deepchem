"""
Gathers all splitters in one place for convenient imports
"""
# flake8: noqa

# basic splitter
from deepchem.splits.splitters import Splitter
from deepchem.splits.splitters import RandomSplitter
from deepchem.splits.splitters import RandomStratifiedSplitter
from deepchem.splits.splitters import RandomGroupSplitter
from deepchem.splits.splitters import SingletaskStratifiedSplitter
from deepchem.splits.splitters import IndexSplitter
from deepchem.splits.splitters import SpecifiedSplitter

# molecule splitter
from deepchem.splits.splitters import ScaffoldSplitter
from deepchem.splits.splitters import MolecularWeightSplitter
from deepchem.splits.splitters import MaxMinSplitter
from deepchem.splits.splitters import FingerprintSplitter
from deepchem.splits.splitters import ButinaSplitter

# other splitter
from deepchem.splits.task_splitter import merge_fold_datasets
from deepchem.splits.task_splitter import TaskSplitter

#################################################################
# Removed API
#################################################################

import logging
logger = logging.getLogger(__name__)


class IndiceSplitter:

  def __init__(self, valid_indices=None, test_indices=None):
    raise ImportError("IndiceSplitter was renamed to SpecifiedSplitter.\n"
                      "Please use SpecifiedSplitter instead of IndiceSplitter.")

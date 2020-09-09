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
from deepchem.splits.splitters import IndiceSplitter

# molecule splitter
from deepchem.splits.splitters import ScaffoldSplitter
from deepchem.splits.splitters import MolecularWeightSplitter
from deepchem.splits.splitters import MaxMinSplitter
from deepchem.splits.splitters import FingerprintSplitter
from deepchem.splits.splitters import ButinaSplitter

# other splitter
from deepchem.splits.task_splitter import merge_fold_datasets
from deepchem.splits.task_splitter import TaskSplitter

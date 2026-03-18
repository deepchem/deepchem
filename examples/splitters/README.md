# Splitter Examples

The DeepChem library has a collection of splitters which demonstrate
how to use DeepChem to split chemical and other datasets in
interesting ways. This folder contains a number of examples which
demonstrate the use of splitters on data

DeepChem has a number of different splitters you can use on your data. Here's the current set

```
from deepchem.splits.splitters import RandomGroupSplitter
from deepchem.splits.splitters import RandomStratifiedSplitter
from deepchem.splits.splitters import SingletaskStratifiedSplitter
from deepchem.splits.splitters import MolecularWeightSplitter
from deepchem.splits.splitters import MaxMinSplitter
from deepchem.splits.splitters import RandomSplitter
from deepchem.splits.splitters import IndexSplitter
from deepchem.splits.splitters import IndiceSplitter
from deepchem.splits.splitters import ClusterFps
from deepchem.splits.splitters import ButinaSplitter
from deepchem.splits.splitters import ScaffoldSplitter
from deepchem.splits.splitters import FingerprintSplitter
from deepchem.splits.splitters import SpecifiedSplitter
from deepchem.splits.splitters import FingerprintSplitter
from deepchem.splits.splitters import TimeSplitterPDBbind
from deepchem.splits.task_splitter import TaskSplitter
```

import os
import numpy as np
import deepchem as dc
seed = 123
np.random.seed(seed)
base_dir = os.getcwd()
data_dir = os.path.join(base_dir, "refined_atomconv")
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
d = dc.data.DiskDataset(data_dir)
splitter = dc.splits.RandomSplitter()
train_dataset, test_dataset = splitter.train_test_split(d, train_dir, test_dir)

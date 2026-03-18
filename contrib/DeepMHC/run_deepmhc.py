"""
Script to train DeepMHC model on the BD2013 dataset.
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from bd13_datasets import aa_charset, load_bd2013_human
from deepmhc import DeepMHC

import numpy as np
import deepchem as dc
from scipy.stats import spearmanr

# Args
pad_len = 13
num_amino_acids = len(aa_charset)
mhc_allele = "HLA-A*02:01"
dropout_p = 0.5
batch_size = 64
nb_epochs = 100

tasks, datasets, transformers = load_bd2013_human(
    mhc_allele=mhc_allele, seq_len=9, pad_len=pad_len)
metric = dc.metrics.Metric(metric=dc.metrics.pearsonr, mode="regression")

train_dataset, _, test_dataset = datasets
model = DeepMHC(num_amino_acids=num_amino_acids, pad_length=pad_len)

model.fit(train_dataset, nb_epoch=nb_epochs)

print("Evaluating models...")
train_scores = model.evaluate(train_dataset, [metric], transformers)
test_scores = model.evaluate(test_dataset, [metric], transformers)

print("Train scores")
print(train_scores['pearsonr'])

print("Test scores")
print(test_scores['pearsonr'])

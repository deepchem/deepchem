"""
Script that trains Atomic Conv models on PDBbind dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import deepchem as dc
import numpy as np
from deepchem.molnet import load_pdbbind
from deepchem.feat import AtomicConvFeaturizer

# For stable runs
np.random.seed(123)

frag1_num_atoms = 70  # for ligand atoms
frag2_num_atoms = 24000  # for protein atoms
complex_num_atoms = frag1_num_atoms + frag2_num_atoms
max_num_neighbors = 12

acf = AtomicConvFeaturizer(
    frag1_num_atoms=frag1_num_atoms,
    frag2_num_atoms=frag2_num_atoms,
    complex_num_atoms=complex_num_atoms,
    max_num_neighbors=max_num_neighbors,
    neighbor_cutoff=4)

pdbbind_tasks, pdbbind_datasets, transformers = load_pdbbind(
    featurizer=acf, split="random", subset="core")
train_dataset, valid_dataset, test_dataset = pdbbind_datasets

metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

model = dc.models.AtomicConvModel(
    n_tasks=len(pdbbind_tasks),
    frag1_num_atoms=frag1_num_atoms,
    frag2_num_atoms=frag2_num_atoms,
    complex_num_atoms=complex_num_atoms)

# Fit trained model
print("Fitting model on train dataset")
model.fit(train_dataset)
# TODO The below line should be fixes
# See: https://github.com/deepchem/deepchem/issues/2373
# model.save()

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

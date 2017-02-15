"""
Script that trains Sklearn singletask models on GDB7 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem as dc
import numpy as np
import shutil
from sklearn.kernel_ridge import KernelRidge

np.random.seed(123)

base_dir = "/tmp/gdb7_sklearn"
data_dir = os.path.join(base_dir, "dataset")
model_dir = os.path.join(base_dir, "model")
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
if os.path.exists(base_dir):
  shutil.rmtree(base_dir)
os.makedirs(base_dir)

max_num_atoms = 23
featurizers = dc.feat.CoulombMatrixEig(max_num_atoms)
input_file = "gdb7.sdf"
tasks = ["u0_atom"]
smiles_field = "smiles"
mol_field = "mol"

featurizer = dc.data.SDFLoader(tasks, smiles_field=smiles_field, mol_field=mol_field, featurizer=featurizers)
dataset = featurizer.featurize(input_file, data_dir)
random_splitter = dc.splits.RandomSplitter()
train_dataset, test_dataset = random_splitter.train_test_split(dataset, train_dir, test_dir, frac_train=0.8)
#transformers = [dc.trans.NormalizationTransformer(transform_X=True, dataset=train_dataset), dc.trans.NormalizationTransformer(transform_y=True, dataset=train_dataset)]
transformers = [dc.trans.NormalizationTransformer(transform_y=True, dataset=train_dataset)]

for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
for transformer in transformers:
    test_dataset = transformer.transform(test_dataset)

regression_metric = dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression")

def model_builder(model_dir):
  sklearn_model = KernelRidge(
      kernel="rbf", alpha=5e-4, gamma=0.008)
  return dc.models.SklearnModel(sklearn_model, model_dir)
model = dc.models.SingletaskToMultitask(tasks, model_builder, model_dir)

# Fit trained model
model.fit(train_dataset)
model.save()

train_evaluator = dc.utils.evaluate.Evaluator(model, train_dataset, transformers)
train_scores = train_evaluator.compute_model_performance([regression_metric])

print("Train scores [kcal/mol]")
print(train_scores)

test_evaluator = dc.utils.evaluate.Evaluator(model, test_dataset, transformers)
test_scores = test_evaluator.compute_model_performance([regression_metric])

print("Validation scores [kcal/mol]")
print(test_scores)

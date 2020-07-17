"""
Script that loads random forest models trained on the sider and tox21 datasets,
predicts on sweetlead, creates covariance matrix

@Author Aneesh Pappu
"""
import os
import sys
import numpy as np
import pandas as pd
import deepchem as dc
from sklearn.ensemble import RandomForestClassifier
from deepchem.models.multitask import SingletaskToMultitask
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.models.sklearn_models import SklearnModel

tox_tasks, (tox_train, tox_valid,
            tox_test), tox_transformers = dc.molnet.load_tox21()

classification_metric = Metric(
    metrics.roc_auc_score, np.mean, mode="classification")


def model_builder(model_dir):
  sklearn_model = RandomForestClassifier(
      class_weight="balanced", n_estimators=500, n_jobs=-1)
  return dc.models.SklearnModel(sklearn_model, model_dir)


print(tox_train.get_task_names())
print(tox_tasks)
tox_model = SingletaskToMultitask(tox_tasks, model_builder)
tox_model.fit(tox_train)

# Load sider models now

sider_tasks, (
    sider_train, sider_valid,
    sider_test), sider_transformers = dc.molnet.load_sider(split="random")

sider_model = SingletaskToMultitask(sider_tasks, model_builder)
sider_model.fit(sider_train)

# Load sweetlead dataset now. Pass in dataset object and appropriate
# transformers to predict functions

sweet_tasks, (sweet_dataset, _, _), sweet_transformers = dc.molnet.load_sweet()

sider_predictions = sider_model.predict(sweet_dataset, sweet_transformers)

tox_predictions = tox_model.predict(sweet_dataset, sweet_transformers)

sider_dimensions = sider_predictions.shape[1]
tox_dimensions = tox_predictions.shape[1]

confusion_matrix = np.zeros(shape=(tox_dimensions, sider_dimensions))
for i in range(tox_predictions.shape[0]):
  nonzero_tox = np.nonzero(tox_predictions[i, :])
  nonzero_sider = np.nonzero(sider_predictions[i, :])
  for j in nonzero_tox[0]:
    for k in nonzero_sider[0]:
      confusion_matrix[j, k] += 1

df = pd.DataFrame(confusion_matrix)

df.to_csv("./tox_sider_matrix.csv")

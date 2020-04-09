import os
import numpy as np
np.random.seed(123)
from sklearn import svm
import tensorflow as tf
tf.random.set_seed(123)
import deepchem as dc
from sklearn.ensemble import RandomForestRegressor
from deepchem.models import GraphConvModel

MODEL = "GraphConv"
SPLIT = "scaffold"
DATASET_NAME = "clearance"


BATCH_SIZE = 128
# Set to higher values to get better numbers
MAX_EPOCH = 10


def load_dataset(dataset_name, featurizer='ECFP', split='index'):
  if dataset_name.lower() == "clearance":
    return dc.molnet.load_clearance(featurizer=featurizer, split=split)
  elif dataset_name.lower() == "hppb":
    return dc.molnet.load_hppb(featurizer=featurizer, split=split)
  elif dataset_name.lower() == "lipo":
    return dc.molnet.load_lipo(featurizer=featurizer, split=split)


def experiment(dataset_name, method='GraphConv', split='scaffold'):
  featurizer = 'ECFP'
  if method == 'GraphConv':
    featurizer = 'GraphConv'
  tasks, datasets, transformers = load_dataset(
      dataset_name, featurizer=featurizer, split=split)
  train, val, test = datasets

  model = None
  if method == 'GraphConv':
    model = GraphConvModel(len(tasks), batch_size=BATCH_SIZE, mode="regression")
  elif method == 'RF':

    def model_builder_rf(model_dir):
      sklearn_model = RandomForestRegressor(n_estimators=100)
      return dc.models.SklearnModel(sklearn_model, model_dir)

    model = dc.models.SingletaskToMultitask(tasks, model_builder_rf)
  elif method == 'SVR':

    def model_builder_svr(model_dir):
      sklearn_model = svm.SVR(kernel='linear')
      return dc.models.SklearnModel(sklearn_model, model_dir)

    model = dc.models.SingletaskToMultitask(tasks, model_builder_svr)

  return model, train, val, test, transformers


#======================================================================
# Run Benchmarks {GC-DNN, SVR, RF}
def main():
  metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

  print("About to build model")
  model, train, val, test, transformers = experiment(
      DATASET_NAME, method=MODEL, split=SPLIT)
  if MODEL == 'GraphConv':
    print("running GraphConv search")
    best_val_score = 0.0
    train_score = 0.0
    for l in range(0, MAX_EPOCH):
      print("epoch %d" % l)
      model.fit(train, nb_epoch=1)
      latest_train_score = model.evaluate(train, [metric],
                                          transformers)['mean-pearson_r2_score']
      latest_val_score = model.evaluate(val, [metric],
                                        transformers)['mean-pearson_r2_score']
      if latest_val_score > best_val_score:
        best_val_score = latest_val_score
        train_score = latest_train_score
    print((MODEL, SPLIT, DATASET_NAME, train_score, best_val_score))
  else:
    model.fit(train)
    train_score = model.evaluate(train, [metric],
                                 transformers)['mean-pearson_r2_score']
    val_score = model.evaluate(val, [metric],
                               transformers)['mean-pearson_r2_score']
    print((MODEL, SPLIT, DATASET_NAME, train_score, val_score))


if __name__ == "__main__":
  main()

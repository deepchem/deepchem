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
DATASET_NAME = "hppb"


BATCH_SIZE = 128
# Set to higher values to get better numbers
MAX_EPOCH = 1

#def retrieve_datasets():
#  os.system(
#      'wget -c %s' %
#      'https://s3-us-west-1.amazonaws.com/deep-crystal-california/az_logd.csv')
#  os.system(
#      'wget -c %s' %
#      'https://s3-us-west-1.amazonaws.com/deep-crystal-california/az_hppb.csv')
#  os.system(
#      'wget -c %s' %
#      'https://s3-us-west-1.amazonaws.com/deep-crystal-california/az_clearance.csv'
#  )


def load_dataset(dataset_name, featurizer='ECFP', split='index'):
  #tasks = ['exp']

  #if featurizer == 'ECFP':
  #  featurizer = dc.feat.CircularFingerprint(size=1024)
  #elif featurizer == 'GraphConv':
  #  featurizer = dc.feat.ConvMolFeaturizer()

  #loader = dc.data.CSVLoader(
  #    tasks=tasks, smiles_field="smiles", featurizer=featurizer)
  #dataset = loader.featurize(dataset_file, shard_size=8192)
  if dataset_name.lower() == "clearance":
    dataset = dc.molnet.load_clearance(featurizer=featurizer, split=split)
  elif dataset_name.lower() == "hppb":
    dataset = dc.molnet.load_hppb(featurizer=featurizer, split=split)

  #transformers = [
  #    dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset)
  #]
  #for transformer in transformers:
  #  dataset = transformer.transform(dataset)

  #splitters = {
  #    'index': dc.splits.IndexSplitter(),
  #    'random': dc.splits.RandomSplitter(),
  #    'scaffold': dc.splits.ScaffoldSplitter()
  #}
  #splitter = splitters[split]
  #train, valid, test = splitter.train_valid_test_split(dataset)
  #return tasks, (train, valid, test), transformers


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
  #print("About to retrieve datasets")
  #retrieve_datasets()

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

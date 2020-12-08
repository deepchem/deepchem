import unittest


class TestDeepchemBuild(unittest.TestCase):

  def setUp(self):
    pass

  def tearDown(self):
    import deepchem
    import os
    import shutil
    data_dir = deepchem.utils.data_utils.get_data_dir()
    bace_dir = os.path.join(data_dir, "bace_c")
    delaney_dir = os.path.join(data_dir, "delaney")
    try:
      shutil.rmtree(bace_dir, ignore_errors=True)
    except:
      pass

  def test_dc_import(self):
    import deepchem
    print(deepchem.__version__)

  def test_rdkit_import(self):
    import rdkit
    print(rdkit.__version__)

  def test_numpy_import(self):
    import numpy as np
    print(np.__version__)

  def test_pandas_import(self):
    import pandas as pd
    print(pd.__version__)

  def get_dataset(self,
                  mode='classification',
                  featurizer='GraphConv',
                  num_tasks=2):
    from deepchem.molnet import load_bace_classification, load_delaney
    import numpy as np
    import deepchem as dc
    from deepchem.data import NumpyDataset
    data_points = 10
    if mode == 'classification':
      tasks, all_dataset, transformers = load_bace_classification(featurizer)
    else:
      tasks, all_dataset, transformers = load_delaney(featurizer)

    train, valid, test = all_dataset
    for i in range(1, num_tasks):
      tasks.append("random_task")
    w = np.ones(shape=(data_points, len(tasks)))

    if mode == 'classification':
      y = np.random.randint(0, 2, size=(data_points, len(tasks)))
      metric = dc.metrics.Metric(
          dc.metrics.roc_auc_score, np.mean, mode="classification")
    else:
      y = np.random.normal(size=(data_points, len(tasks)))
      metric = dc.metrics.Metric(
          dc.metrics.mean_absolute_error, mode="regression")

    ds = NumpyDataset(train.X[:data_points], y, w, train.ids[:data_points])

    return tasks, ds, transformers, metric

  def test_graph_conv_model(self):
    from deepchem.models import GraphConvModel, TensorGraph
    import numpy as np
    tasks, dataset, transformers, metric = self.get_dataset(
        'classification', 'GraphConv')

    batch_size = 50
    model = GraphConvModel(
        len(tasks), batch_size=batch_size, mode='classification')

    model.fit(dataset, nb_epoch=10)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean-roc_auc_score'] >= 0.9


if __name__ == '__main__':
  unittest.main()

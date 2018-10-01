import deepchem
import numpy as np
import tensorflow as tf
import unittest
from nose.plugins.attrib import attr
from deepchem.models.tensorgraph.models import resnet50
from deepchem.models.tensorgraph import layers


class TestResNet50(unittest.TestCase):

  @attr('slow')
  def test_resnet50(self):
    resnet = deepchem.models.tensorgraph.models.resnet50.ResNet50(
        learning_rate=0.003,
        img_rows=128,
        img_cols=128,
        model_dir='./resnet50/')

    # Prepare Training Data
    data = np.ones((1, 128, 128, 3))
    position = np.array([1])
    labels = np.zeros((1, resnet.classes))
    labels[np.arange(1), position] = 1
    train = deepchem.data.NumpyDataset(data, labels)
    # Train the model
    resnet.fit(train, nb_epochs=0)
    resnet.save()
    # Prepare the Testing data
    test_data = np.ones((2, 128, 128, 3))
    test = deepchem.data.NumpyDataset(test_data)
    # predict
    predictions = resnet.predict(test)
    # check output shape
    self.assertEqual(predictions.shape, (2, 1000))

    # new object of ResNet to test if loading the model results in same predictions
    resnet_new = deepchem.models.tensorgraph.models.resnet50.ResNet50(
        learning_rate=0.003,
        img_rows=128,
        img_cols=128,
        model_dir='./resnet50/')
    resnet_new.load_from_dir('./resnet50/')
    resnet_new.restore()
    predictions_new = resnet_new.predict(test)

    self.assertTrue(np.all(predictions == predictions_new))

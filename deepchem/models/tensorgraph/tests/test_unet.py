import deepchem
import numpy as np
import tensorflow as tf
import unittest
from nose.plugins.attrib import attr
from deepchem.models.tensorgraph.models import unet
from deepchem.models.tensorgraph import layers


class TestUNet(unittest.TestCase):

  @attr('slow')
  def test_unet(self):
    unet2D = deepchem.models.tensorgraph.models.unet.UNet(
        learning_rate=0.003, img_rows=32, img_cols=32, model_dir='./unet/')
    # Prepare Training Data
    data = np.ones((5, 32, 32, 3))
    labels = np.ones((5, 32 * 32))
    train = deepchem.data.NumpyDataset(data, labels)
    # Train the model
    unet2D.fit(train, nb_epochs=2)
    unet2D.save()
    # Prepare the Testing data
    test_data = np.ones((2, 32, 32, 3))
    test = deepchem.data.NumpyDataset(test_data)
    # predict
    predictions = unet2D.predict(test)
    # check output shape
    self.assertEqual(predictions.shape, (2, 32, 32, 1))

    # new object of UNet to test if loading the model results in same predictions
    unet2D_new = deepchem.models.tensorgraph.models.unet.UNet(
        learning_rate=0.003, img_rows=32, img_cols=32, model_dir='./unet/')
    unet2D_new.load_from_dir('./unet/')
    unet2D_new.restore()
    predictions_new = unet2D_new.predict(test)

    self.assertTrue(np.all(predictions == predictions_new))

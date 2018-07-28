import deepchem as dc
import numpy as np
import tensorflow as tf
import unittest
from deepchem.models.tensorgraph.models import unet
from deepchem.models.tensorgraph import layers


class TestUNet(unittest.TestCase):

  def test_unet(self):
    unet2D = deepchem.models.tensorgraph.models.unet.UNet(
        learning_rate=0.003, img_rows=32, img_cols=32)
    data = np.ones((5, 32, 32, 3))
    labels = np.ones((5, 32 * 32))
    train = deepchem.data.NumpyDataset(data, labels)
    unet2D.fit(train, nb_epochs=0)

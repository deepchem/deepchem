"""
Code for processing the Google vs-datasets using keras.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
from keras.models import Graph
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization 
from keras.optimizers import SGD
from deepchem.models import Model

class KerasModel(Model):
  """
  Abstract base class shared across all Keras models.
  """

  def save(self):
    """
    Saves underlying keras model to disk.
    """
    super(KerasModel, self).save()
    model = self.get_raw_model()
    filename, _ = os.path.splitext(Model.get_model_filename(self.model_dir))

    # Note that keras requires the model architecture and weights to be stored
    # separately. A json file is generated that specifies the model architecture.
    # The weights will be stored in an h5 file. The pkl.gz file with store the
    # target name.
    json_filename = "%s.%s" % (filename, "json")
    h5_filename = "%s.%s" % (filename, "h5")
    # Save architecture
    json_string = model.to_json()
    with open(json_filename, "wb") as file_obj:
      file_obj.write(json_string)
    model.save_weights(h5_filename, overwrite=True)

  def load(self, model_dir):
    """
    Load keras multitask DNN from disk.
    """
    filename = Model.get_model_filename(model_dir)
    filename, _ = os.path.splitext(filename)

    json_filename = "%s.%s" % (filename, "json")
    h5_filename = "%s.%s" % (filename, "h5")

    with open(json_filename) as file_obj:
      model = model_from_json(file_obj.read())
    model.load_weights(h5_filename)
    self.raw_model = model


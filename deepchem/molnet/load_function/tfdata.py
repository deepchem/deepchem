from __future__ import division
from __future__ import unicode_literals

import logging
import tensorflow as tf

logger = logging.getLogger(__name__)


class Tfdata(object):

  def __init__(self, dataset):
    for i in range(dataset.metadata_df.shape[0]):
      self.X_tfdata = tf.data.Dataset.from_tensor_slices(
          dataset.get_shard(i)[0])
      self.y_tfdata = tf.data.Dataset.from_tensor_slices(
          dataset.get_shard(i)[1])
      self.w_tfdata = tf.data.Dataset.from_tensor_slices(
          dataset.get_shard(i)[2])
      self.ibs_tfdata = tf.data.Dataset.from_tensor_slices(
          dataset.get_shard(i)[3])

  def make_one_shot_iterator(self):
    return (self.X_tfdata.make_one_shot_iterator().get_next(),
            self.y_tfdata.make_one_shot_iterator().get_next(),
            self.ibs_tfdata.make_one_shot_iterator().get_next(),
            self.ibs_tfdata.make_one_shot_iterator().get_next())

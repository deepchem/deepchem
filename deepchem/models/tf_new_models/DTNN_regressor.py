import tensorflow as tf
from deepchem.models.tf_new_models.multitask_regressor import MultitaskGraphRegressor


class DTNNGraphRegressor(MultitaskGraphRegressor):

  def build(self):
    # Create target inputs
    self.label_placeholder = tf.placeholder(
        dtype='float32', shape=(None, self.n_tasks), name="label_placeholder")
    self.weight_placeholder = tf.placeholder(
        dtype='float32', shape=(None, self.n_tasks), name="weight_placholder")

    feat = self.model.return_outputs()
    feat_size = self.feat_dim
    # dimension of `feat` becomes Unknown after tf.tensordot operation
    # need to define dimension of W and b explicitly
    outputs = []
    W_list = []
    b_list = []
    for task in range(self.n_tasks):
      W_list.append(
          tf.Variable(
              tf.truncated_normal([feat_size, 1], stddev=0.01),
              name='w',
              dtype=tf.float32))
      b_list.append(tf.Variable(tf.zeros([1]), name='b', dtype=tf.float32))
      outputs.append(
          tf.squeeze(tf.nn.xw_plus_b(feat, W_list[task], b_list[task])))
    return outputs

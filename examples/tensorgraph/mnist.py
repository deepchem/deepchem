from tensorflow.examples.tutorials.mnist import input_data

from deepchem.models import TensorGraph

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import deepchem as dc
import tensorflow as tf
from deepchem.models.tensorgraph.layers import Layer, Input, Reshape, Flatten
from deepchem.models.tensorgraph.layers import Dense, SoftMaxCrossEntropy, ReduceMean, SoftMax

train = dc.data.NumpyDataset(mnist.train.images, mnist.train.labels)
valid = dc.data.NumpyDataset(mnist.validation.images, mnist.validation.labels)

tg = dc.models.TensorGraph(
    tensorboard=True, model_dir='/tmp/mnist', batch_size=1000)
feature = Input(shape=(None, 784))
tg.add_feature(feature)

# Images are square 28x28 (batch, height, width, channel)
make_image = Reshape(shape=(-1, 28, 28, 1))
tg.add_layer(make_image, parents=[feature])


class Conv2d(Layer):

  def __init__(self, num_outputs, kernel_size=5, **kwargs):
    self.num_outputs = num_outputs
    self.kernel_size = kernel_size
    super().__init__(**kwargs)

  def __call__(self, *parents):
    parent_tensor = parents[0].out_tensor
    out_tensor = tf.contrib.layers.conv2d(
        parent_tensor,
        num_outputs=self.num_outputs,
        kernel_size=self.kernel_size,
        padding="SAME",
        activation_fn=tf.nn.relu,
        normalizer_fn=tf.contrib.layers.batch_norm)
    self.out_tensor = tf.nn.max_pool(
        out_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return self.out_tensor


conv2d_1 = Conv2d(num_outputs=32)
tg.add_layer(conv2d_1, parents=[make_image])

conv2d_2 = Conv2d(num_outputs=64)
tg.add_layer(conv2d_2, parents=[conv2d_1])

flatten = Flatten()
tg.add_layer(flatten, parents=[conv2d_2])

dense1 = Dense(out_channels=1024, activation_fn=tf.nn.relu)
tg.add_layer(dense1, parents=[flatten])

dense2 = Dense(out_channels=10)
tg.add_layer(dense2, parents=[dense1])

label = Input(shape=(None, 10))
tg.add_label(label)

smce = SoftMaxCrossEntropy()
tg.add_layer(smce, parents=[label, dense2])

loss = ReduceMean()
tg.add_layer(loss, parents=[smce])
tg.set_loss(loss)

output = SoftMax()
tg.add_layer(output, parents=[dense2])
tg.add_output(output)

tg.fit(train, nb_epoch=2)
tg.fit(train, nb_epoch=2)
tg.save()

# tg = TensorGraph.load_from_dir("/tmp/mnist")
from sklearn.metrics import roc_curve, auc
import numpy as np

print("Validation")
prediction = np.squeeze(tg.predict_on_batch(valid.X))
print(prediction[0])
print(valid.y[0])

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(10):
  l = valid.y[:, i]
  print(l.shape)
  fpr[i], tpr[i], thresh = roc_curve(valid.y[:, i], prediction[:, i])
  roc_auc[i] = auc(fpr[i], tpr[i])
  print("%s:%s" % (i, roc_auc[i]))

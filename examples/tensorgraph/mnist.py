from tensorflow.examples.tutorials.mnist import input_data

from deepchem.models import TensorGraph

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import deepchem as dc
import tensorflow as tf
from deepchem.models.tensorgraph.layers import Layer, Input, Reshape, Flatten, Feature, Conv2d, MaxPool, Label
from deepchem.models.tensorgraph.layers import Dense, SoftMaxCrossEntropy, ReduceMean, SoftMax

train = dc.data.NumpyDataset(mnist.train.images, mnist.train.labels)
valid = dc.data.NumpyDataset(mnist.validation.images, mnist.validation.labels)

# Images are square 28x28 (batch, height, width, channel)
feature = Feature(shape=(None, 784), name="Feature")
make_image = Reshape(shape=(-1, 28, 28, 1), in_layers=[feature])

conv2d_1 = Conv2d(num_outputs=32, in_layers=[make_image])
maxpool_1 = MaxPool(in_layers=[conv2d_1])

conv2d_2 = Conv2d(num_outputs=64, in_layers=[maxpool_1])
maxpool_2 = MaxPool(in_layers=[conv2d_2])
flatten = Flatten(in_layers=[maxpool_2])

dense1 = Dense(out_channels=1024, activation_fn=tf.nn.relu, in_layers=[flatten])
dense2 = Dense(out_channels=10, in_layers=[dense1])
label = Label(shape=(None, 10), name="Label")
smce = SoftMaxCrossEntropy(in_layers=[label, dense2])
loss = ReduceMean(in_layers=[smce])
output = SoftMax(in_layers=[dense2])

tg = dc.models.TensorGraph(
    tensorboard=True, model_dir='/tmp/mnist', batch_size=1000, use_queue=True)
tg.add_output(output)
tg.set_loss(loss)
tg.fit(train, nb_epoch=2)
tg.fit(train, nb_epoch=2)

from sklearn.metrics import roc_curve, auc
import numpy as np

print("Validation")
prediction = np.squeeze(tg.predict_proba_on_batch(valid.X))
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

from nose.tools import timed
@timed(180)
def test_notebook():
  
  # coding: utf-8
  
  # # MNIST with DeepChem and TensorGraph
  
  # In[1]:
  
  
  from tensorflow.examples.tutorials.mnist import input_data
  
  
  # In[2]:
  
  
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  
  
  # In[3]:
  
  
  import deepchem as dc
  import tensorflow as tf
  from deepchem.models.tensorgraph.layers import Layer, Input, Reshape, Flatten, Conv2D, Label, Feature
  from deepchem.models.tensorgraph.layers import Dense, SoftMaxCrossEntropy, ReduceMean, SoftMax
  
  
  # In[4]:
  
  
  train = dc.data.NumpyDataset(mnist.train.images, mnist.train.labels)
  valid = dc.data.NumpyDataset(mnist.validation.images, mnist.validation.labels)
  
  
  # In[5]:
  
  
  tg = dc.models.TensorGraph(tensorboard=True, model_dir='/tmp/mnist', use_queue=False)
  feature = Feature(shape=(None, 784))
  
  # Images are square 28x28 (batch, height, width, channel)
  make_image = Reshape(shape=(-1, 28, 28, 1), in_layers=[feature])
  
  conv2d_1 = Conv2D(num_outputs=32, in_layers=[make_image])
  
  conv2d_2 = Conv2D(num_outputs=64, in_layers=[conv2d_1])
  
  flatten = Flatten(in_layers=[conv2d_2])
  
  dense1 = Dense(out_channels=1024, activation_fn=tf.nn.relu, in_layers=[flatten])
  
  dense2 = Dense(out_channels=10, in_layers=[dense1])
  
  label = Label(shape=(None, 10))
  
  smce = SoftMaxCrossEntropy(in_layers=[label, dense2])
  loss = ReduceMean(in_layers=[smce])
  tg.set_loss(loss)
  
  output = SoftMax(in_layers=[dense2])
  tg.add_output(output)
  
  
  # In[6]:
  
  
  # nb_epoch set to 0 to permit rendering of tutorials online.
  # Set nb_epoch=10 for better results
  tg.fit(train, nb_epoch=0)
  
  
  # In[7]:
  
  
  # Note that AUCs will be nonsensical without setting nb_epoch higher!
  from sklearn.metrics import roc_curve, auc
  import numpy as np
  
  print("Validation")
  prediction = np.squeeze(tg.predict_on_batch(valid.X))
  
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(10):
      fpr[i], tpr[i], thresh = roc_curve(valid.y[:, i], prediction[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])
      print("class %s:auc=%s" % (i, roc_auc[i]))
  

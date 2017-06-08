

```python
from tensorflow.examples.tutorials.mnist import input_data
```


```python
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz



```python
import deepchem as dc
import tensorflow as tf
from deepchem.models.tensorgraph.layers import Layer, Input, Reshape, Flatten
from deepchem.models.tensorgraph.layers import Dense, SoftMaxCrossEntropy, ReduceMean, SoftMax
```


```python
train = dc.data.NumpyDataset(mnist.train.images, mnist.train.labels)
valid = dc.data.NumpyDataset(mnist.validation.images, mnist.validation.labels)
```


```python
tg = dc.models.TensorGraph(tensorboard=True, model_dir='/tmp/mnist')
feature = Input(shape=(None, 784))
tg.add_layer(feature)
tg.add_feature(feature)

# Images are square 28x28 (batch, height, width, channel)
make_image = Reshape(shape=(-1, 28, 28, 1))
tg.add_layer(make_image, parents=[feature])

class Conv2d(Layer):
    def __init__(self, num_outputs, kernel_size=5, **kwargs):
        self.num_outputs = num_outputs
        self.kernel_size = kernel_size
        super(Conv2d, self).__init__(**kwargs)
    def __call__(self, *parents):
        parent_tensor = parents[0].out_tensor
        out_tensor = tf.contrib.layers.conv2d(parent_tensor,
                                              num_outputs=self.num_outputs,
                                              kernel_size = self.kernel_size,
                                              padding="SAME",
                                              activation_fn=tf.nn.relu,
                                              normalizer_fn=tf.contrib.layers.batch_norm)
        self.out_tensor = tf.nn.max_pool(out_tensor,
                                         ksize=[1, 2, 2, 1],
                                         strides=[1, 2, 2, 1],
                                         padding='SAME')
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
tg.add_layer(label, parents=list())
tg.add_label(label)

smce = SoftMaxCrossEntropy()
tg.add_layer(smce, parents=[label, dense2])

loss = ReduceMean()
tg.add_layer(loss, parents=[smce])
tg.set_loss(loss)

output = SoftMax()
tg.add_layer(output, parents=[dense2])
tg.add_output(output)
```


```python
tg.fit(train, nb_epoch=10)
tg.save()
```

    Training for 10 epochs
    On batch 0
    On batch 50
    On batch 100
    On batch 150
    On batch 200
    On batch 250
    On batch 300
    On batch 350
    On batch 400
    On batch 450
    On batch 500
    On batch 550
    On batch 600
    On batch 650
    On batch 700
    On batch 750
    On batch 800
    On batch 850
    On batch 900
    On batch 950
    On batch 1000
    On batch 1050
    Ending epoch 0: Average loss 0.198305
    On batch 0
    On batch 50
    On batch 100
    On batch 150
    On batch 200
    On batch 250
    On batch 300
    On batch 350
    On batch 400
    On batch 450
    On batch 500
    On batch 550
    On batch 600
    On batch 650
    On batch 700
    On batch 750
    On batch 800
    On batch 850
    On batch 900
    On batch 950
    On batch 1000
    On batch 1050
    Ending epoch 1: Average loss 0.0536864
    On batch 0
    On batch 50
    On batch 100
    On batch 150
    On batch 200
    On batch 250
    On batch 300
    On batch 350
    On batch 400
    On batch 450
    On batch 500
    On batch 550
    On batch 600
    On batch 650
    On batch 700
    On batch 750
    On batch 800
    On batch 850
    On batch 900
    On batch 950
    On batch 1000
    On batch 1050
    Ending epoch 2: Average loss 0.0425871
    On batch 0
    On batch 50
    On batch 100
    On batch 150
    On batch 200
    On batch 250
    On batch 300
    On batch 350
    On batch 400
    On batch 450
    On batch 500
    On batch 550
    On batch 600
    On batch 650
    On batch 700
    On batch 750
    On batch 800
    On batch 850
    On batch 900
    On batch 950
    On batch 1000
    On batch 1050
    Ending epoch 3: Average loss 0.0336403
    On batch 0
    On batch 50
    On batch 100
    On batch 150
    On batch 200
    On batch 250
    On batch 300
    On batch 350
    On batch 400
    On batch 450
    On batch 500
    On batch 550
    On batch 600
    On batch 650
    On batch 700
    On batch 750
    On batch 800
    On batch 850
    On batch 900
    On batch 950
    On batch 1000
    On batch 1050
    Ending epoch 4: Average loss 0.028042
    On batch 0
    On batch 50
    On batch 100
    On batch 150
    On batch 200
    On batch 250
    On batch 300
    On batch 350
    On batch 400
    On batch 450
    On batch 500
    On batch 550
    On batch 600
    On batch 650
    On batch 700
    On batch 750
    On batch 800
    On batch 850
    On batch 900
    On batch 950
    On batch 1000
    On batch 1050
    Ending epoch 5: Average loss 0.0257894
    On batch 0
    On batch 50
    On batch 100
    On batch 150
    On batch 200
    On batch 250
    On batch 300
    On batch 350
    On batch 400
    On batch 450
    On batch 500
    On batch 550
    On batch 600
    On batch 650
    On batch 700
    On batch 750
    On batch 800
    On batch 850
    On batch 900
    On batch 950
    On batch 1000
    On batch 1050
    Ending epoch 6: Average loss 0.0187417
    On batch 0
    On batch 50
    On batch 100
    On batch 150
    On batch 200
    On batch 250
    On batch 300
    On batch 350
    On batch 400
    On batch 450
    On batch 500
    On batch 550
    On batch 600
    On batch 650
    On batch 700
    On batch 750
    On batch 800
    On batch 850
    On batch 900
    On batch 950
    On batch 1000
    On batch 1050
    Ending epoch 7: Average loss 0.01757
    On batch 0
    On batch 50
    On batch 100
    On batch 150
    On batch 200
    On batch 250
    On batch 300
    On batch 350
    On batch 400
    On batch 450
    On batch 500
    On batch 550
    On batch 600
    On batch 650
    On batch 700
    On batch 750
    On batch 800
    On batch 850
    On batch 900
    On batch 950
    On batch 1000
    On batch 1050
    Ending epoch 8: Average loss 0.0132135
    On batch 0
    On batch 50
    On batch 100
    On batch 150
    On batch 200
    On batch 250
    On batch 300
    On batch 350
    On batch 400
    On batch 450
    On batch 500
    On batch 550
    On batch 600
    On batch 650
    On batch 700
    On batch 750
    On batch 800
    On batch 850
    On batch 900
    On batch 950
    On batch 1000
    On batch 1050
    Loggin
    Ending epoch 9: Average loss 0.0121854
    TIMING: model fitting took 77.903 s



```python
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
```

    Validation
    class 0:auc=0.99987578265
    class 1:auc=0.999975981083
    class 2:auc=0.99987010958
    class 3:auc=0.999980197583
    class 4:auc=0.999985348139
    class 5:auc=0.999889990331
    class 6:auc=0.999762644083
    class 7:auc=0.999936670072
    class 8:auc=0.999936085657
    class 9:auc=0.999968609514



```python

```


```python

```

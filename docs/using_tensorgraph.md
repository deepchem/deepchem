# Using TensorGraph

Tensorgraph is the newest version of DeepChem's Tensorflow model building framework.
Tensorgraph is designed to be incredibly fexible for users attempting research, but also easy to use for Deep Learning begginners.

## Vocabulary

### TensorGraph
A TensorGraph model is pipeing around a series of layers.  Placing your Tensorflow code in a Tensorgraph means you don't have to worry about code to fit, train, batch data, write pickling function, write restoring functions, and enable beautiful tensorboard visualizations.

### Layer
A Layer is a node in a graph.  A Layer class has a "name" and a __call__ function.  __call__ takes in a list of parent Layers, and returns a tf.Tensor object, and sets the tf.Tensor object as self.out_tensor.  Below is an example layer which will reverse the order of the parent Tensor 0th axis.

``` python
class Reverse(Layer):
  def __call__(self, *parents):
    parent_out = parents[0].out_tensor
    out_tensor = tf.reverse(parent_out, axis=0)
    return out_tensor
```
You can combine these layers to form Graph of computation by combining these layers on a tensorgraph.
``` python
tg = TensorGraph()
feature = Input(shape=(None, 5))
tg.add_layer(feature, parents=list())
reverse = Reverse() # From Above
tg.add_layer(Reverse(), parentes=[feature])
```
Now during computation the reverse layer will reverse the feature layer.

### Features
Features are how we feed input data into our model.  During training or predicting Tensorgraph will set the values of feature layers from Dataset.X.  We have to manually tell TensorGraph which layers we want to feed data into.

``` python
tg = TensorGraph()
# Convention is to set batch size as 0th axis.
# None means any batch size is allowed.
feature = Input(shape=(None, 5))
tg.add_layer(feature)
tg.add_feature(feature)
```

### Labels
Labels are the results we expect to get for each feature.  This is what we feed Dataset.y into when "fit"ing a model.
```
tf = TensorGraph()
label = Input(shape=(None, 2)) # Example Binary Classification label
tf.add_layer(label)
tf.add_label(label)
```

### Loss
Loss is a scalar we wish to minimize for our network.  When training we will use tensorflow to intelligently minimize this value via gradient descent.

``` python
tg = TensorGraph()
# a is labels for a classification problem
# b is our guesses for the classification problem
smce = SoftMaxCrossEntropy()
tg.add_layer(smce, parents=[a, b])
redmean = ReduceMean()
tg.add_layer(redmean, parents=[smce])
tg.set_loss(redmean)
```
Here in our classification problem we are attempting to reduce the mean of a SoftMaxCrossEntropy for a classification problem.

### Outputs
Outputs are your predictions for tasks.  A TensorGraph can have multiple outputs!  An example is a multi-task regression problem.
By convention for regression problems outputs are single values. Classification problems outputs are a probability vector,
where each entry represents the probability of the input beint in each of the classes.

``` python
tg = TensorGraph()
# Create your layers
# a is an output layer
tg.add_output(a)
```

## Creating your own TensorGraph Model
In order to create a a TensorGraph model we have to
1. Create the TensorGraph model object
2. Add Layers
3. Add Features
4. Add Labels
5. Set Loss

## Examples
TensorGraph on the MNIST

TensorGraph for Bypass MultiTask Classification

## Further Reading
Using Tensorboard with TensorGraph

Saving and Restoring TensorGraph

Experimenting on new architectures with TensorGraph

Adding complexity while fighting overfitting

Transfer Learning

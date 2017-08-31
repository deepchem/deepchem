# Using TensorGraph

Tensorgraph is the newest version of DeepChem's Tensorflow model building framework.
Tensorgraph is designed to be incredibly flexible for users attempting research as well as being easy to learn for Deep Learning beginners.

## Vocabulary

### TensorGraph
A TensorGraph model is pipeing around a series of layers.  Placing your Tensorflow code into a Tensorgraph means you don't have to worry about coding methods for fitting, training, and batching data, nor do you need to worry about writing picking and restoring functions. It also enables beautiful Tensorboard visualizations. 

### Layer
A Layer is a node in a graph.  A Layer class has a "name" and a create_tensor function.  create_tensor takes in a list of input Layers, returns a tf.Tensor object, and sets the tf.Tensor object as self.out_tensor.  Below is an example layer which will reverse the order of the parent Tensor's 0th axis.

``` python
class Reverse(Layer):
  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    parent_out = in_layers[0].out_tensor
    out_tensor = tf.reverse(parent_out, axis=0)
    if set_tensors:
    	self.out_tensor = out_tensor
    return out_tensor
```
You can combine these layers to form a Graph of computation by combining these layers on a tensorgraph.
``` python
tg = TensorGraph()
feature = Input(shape=(None, 5))
reverse = Reverse(in_layers=[feature]) # From Above
```
Now during computation, the reverse layer will reverse the feature layer.

### Features
Features are how we feed input data into our model.  During training or predicting, Tensorgraph will set the values of feature layers from Dataset.X.  We have to manually tell TensorGraph which layers we want to feed data into.

``` python
tg = TensorGraph()
# Convention is to set batch size as 0th axis.
# None means any batch size is allowed.
feature = Feature(shape=(None, 5))
```

### Labels
Labels are the results we expect to get for each feature.  We feed Dataset.y into a label when "fit"ing a model.
```
tf = TensorGraph()
label = Label(shape=(None, 2)) # Example Binary Classification label
```

### Loss
Loss is a scalar we wish to minimize for our network.  When training, we will use tensorflow to intelligently minimize this value via gradient descent.

``` python
tg = TensorGraph()
# a is labels for a classification problem
# b is our guesses for the classification problem
smce = SoftMaxCrossEntropy(in_layers=[a,b])
redmean = ReduceMean(in_layers=[smce])
tg.set_loss(redmean)
```
Here in our classification problem, we are attempting to reduce the mean of a SoftMaxCrossEntropy.

### Outputs
Outputs are your predictions for tasks.  A TensorGraph can have multiple outputs!  An example is a multi-task regression problem.
By convention, regression problems' outputs are single values, while classification problems' outputs are probability vectors,
where each entry represents the probability of the input being in each of the classes.

``` python
tg = TensorGraph()
# Create your layers
# a is an output layer
tg.add_output(a)
```

## Creating your own TensorGraph Model
In order to create a TensorGraph model, we have to:
1. Create the TensorGraph model object
2. Add Layers
3. Add Features
4. Add Labels
5. Set Loss

## Using TensorBoard with TensorGraph
Visualizing layers is extremely easy with Tensorboard. After creating a layer, simply call the set_summary method. 
Tensorgraph currently supports the following summary ops: [histogram](https://www.tensorflow.org/api_docs/python/tf/summary/histogram), [scalar](https://www.tensorflow.org/api_docs/python/tf/summary/scalar), and [tensor_summary](https://www.tensorflow.org/api_docs/python/tf/summary/tensor_summary)

``` python
model_dir='example_dir'
tg = TensorGraph(model_dir=model_dir)
# a is Input tensor
dense = Dense(in_layers=[a])
dense.set_summary(summary_op='histogram')
```
Then after the model has finished training, simply run
``` bash
tensorboard --logdir=example_dir
```
Navigate to: [localhost:6006](localhost:6006) to view your visualizations. Tensorgraph will also have automatically constructed the graph visualization.

## Examples
TensorGraph on the MNIST

TensorGraph for Bypass MultiTask Classification

## Further Reading
Saving and Restoring TensorGraph

Experimenting on new architectures with TensorGraph

Adding complexity while fighting overfitting

Transfer Learning
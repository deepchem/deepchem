def test_notebook():
  
  # coding: utf-8
  
  # # TensorGraph Mechanics
  # In this IPython notebook, we will cover more advanced aspects of the `TensorGraph` framework. In particular, we will demonstrate how to share weights between layers and show how to use `DataBag` to reduce the amount of overhead needed to train complex `TensorGraph` models.
  # 
  # Let's start by defining a `TensorGraph` object.
  
  # In[1]:
  
  
  import deepchem as dc
  from deepchem.models.tensorgraph.tensor_graph import TensorGraph
  
  tg = TensorGraph(use_queue=False)
  
  
  # We're going to construct an architecture that has two identical feature inputs. Let's call these feature inputs `left_features` and `right_features`.
  
  # In[2]:
  
  
  from deepchem.models.tensorgraph.layers import Feature
  
  left_features = Feature(shape=(None, 75))
  right_features = Feature(shape=(None, 75))
  
  
  # Let's now apply a nonlinear transformation to both `left_features` and `right_features`. We can use the `Dense` layer to do so. In addition, let's make sure that we apply the same nonlinear transformation to both `left_features` and `right_features`. To this, we can use the `Layer.shared()`. We use this method by initializing a first `Dense` layer, and then calling the `Layer.shared()` method to make a copy of that layer.
  
  # In[3]:
  
  
  from deepchem.models.tensorgraph.layers import Dense
  
  
  dense_left = Dense(out_channels=1, in_layers=[left_features])
  dense_right = dense_left.shared(in_layers=[right_features])
  
  
  # Let's now combine these two transformed feature layers by addition. We will assume this network is being used to solve a regression problem, so we will introduce a `Label` that stores the true regression values. We can then define the objective function of the network via the `L2Loss` between the added output and the true label.
  
  # In[4]:
  
  
  from deepchem.models.tensorgraph.layers import Add
  from deepchem.models.tensorgraph.layers import Label
  from deepchem.models.tensorgraph.layers import L2Loss
  from deepchem.models.tensorgraph.layers import ReduceMean
  
  output = Add(in_layers=[dense_left, dense_right])
  tg.add_output(output)
  
  labels = Label(shape=(None, 1))
  batch_loss = L2Loss(in_layers=[labels, output])
  # Need to reduce over the loss
  loss = ReduceMean(in_layers=batch_loss)
  tg.set_loss(loss)
  
  
  # Let's now randomly sample an artificial dataset we can use to train this architecture. We will need to sample the `left_features`, `right_features`, and `labels` in order to be able to train the network.
  
  # In[5]:
  
  
  import numpy as np
  import numpy.random
  
  n_samples = 100
  sampled_left_features = np.random.rand(100, 75)
  sampled_right_features = np.random.rand(100, 75)
  sampled_labels = np.random.rand(75, 1)
  
  
  # How can we train `TensorGraph` networks with multiple `Feature` inputs? One option is to manually construct a python generator that provides inputs. The tutorial notebook on graph convolutions does this explicitly. For simpler cases, we can use the convenience object `DataBag` which makes it easier to construct generators. A `DataBag` holds multiple datasets (added via `DataBag.add_dataset`). The method `DataBag.iterbatches()` will construct a generator that peels off batches of the desired size from each dataset and return a dictionary mapping inputs (`Feature`, `Label`, and `Weight` objects) to data for that minibatch. Let's see `DataBag` in action.
  # 
  # Note that we will need to wrap our sampled Numpy arrays with `NumpyDataset` objects for our call to work.
  
  # In[6]:
  
  
  from deepchem.data.datasets import Databag
  from deepchem.data.datasets import NumpyDataset
  
  databag = Databag()
  databag.add_dataset(left_features, NumpyDataset(sampled_left_features))
  databag.add_dataset(right_features, NumpyDataset(sampled_right_features))
  databag.add_dataset(labels, NumpyDataset(sampled_labels))
  
  
  # Let's now train this architecture! We need to use the method `TensorGraph.fit_generator()` passing in a generator created by `databag.iterbatches()`.
  
  # In[7]:
  
  
  tg.fit_generator(
      databag.iterbatches(epochs=100, batch_size=50, pad_batches=True))
  
  
  # You should now be able to construct more sophisticated `TensorGraph` architectures with relative ease!

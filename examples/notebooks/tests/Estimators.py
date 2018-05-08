from nose.tools import timed
@timed(180)
def test_notebook():
  
  # coding: utf-8
  
  # Using DeepChem with Tensorflow Data and Estimators
  # -----------------------------------------------
  # 
  # When DeepChem was first created, Tensorflow had no standard interface for datasets or models.  We created the Dataset and Model classes to fill this hole.  More recently, Tensorflow has added the `tf.data` module as a standard interface for datasets, and the `tf.estimator` module as a standard interface for models.  To enable easy interoperability with other tools, we have added features to Dataset and Model to support these new standards.
  # 
  # This example demonstrates how to use these features.  Let's begin by loading a dataset and creating a model to analyze it.  We'll use a simple MultiTaskClassifier with one hidden layer.
  
  # In[1]:
  
  
  import deepchem as dc
  import tensorflow as tf
  import numpy as np
  
  tasks, datasets, transformers = dc.molnet.load_tox21()
  train_dataset, valid_dataset, test_dataset = datasets
  n_tasks = len(tasks)
  n_features = train_dataset.X.shape[1]
  
  model = dc.models.MultiTaskClassifier(n_tasks, n_features, layer_sizes=[1000], dropouts=0.25)
  
  
  # We want to train the model using the training set, then evaluate it on the test set.  As our evaluation metric we will use the ROC AUC, averaged over the 12 tasks included in the dataset.  First let's see how to do this with the DeepChem API.
  
  # In[2]:
  
  
  model.fit(train_dataset, nb_epoch=1)
  metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)
  print(model.evaluate(test_dataset, [metric]))
  
  
  # Simple enough.  Now let's see how to do the same thing with the Tensorflow APIs.  Fair warning: this is going to take a lot more code!
  # 
  # To begin with, Tensorflow doesn't allow a dataset to be passed directly to a model.  Instead, you need to write an "input function" to construct a particular set of tensors and return them in a particular format.  Fortunately, Dataset's `make_iterator()` method provides exactly the tensors we need in the form of a `tf.data.Iterator`.  This allows our input function to be very simple.
  
  # In[3]:
  
  
  def input_fn(dataset, epochs):
      x, y, weights = dataset.make_iterator(batch_size=100, epochs=epochs).get_next()
      return {'x': x, 'weights': weights}, y
  
  
  # Next, you have to use the functions in the `tf.feature_column` module to create an object representing each feature and weight column (but curiously, *not* the label column—don't ask me why!).  These objects describe the data type and shape of each column, and give each one a name.  The names must match the keys in the dict returned by the input function.
  
  # In[4]:
  
  
  x_col = tf.feature_column.numeric_column('x', shape=(n_features,))
  weight_col = tf.feature_column.numeric_column('weights', shape=(n_tasks,))
  
  
  # Unlike DeepChem models, which allow arbitrary metrics to be passed to `evaluate()`, estimators require all metrics to be defined up front when you create the estimator.  Unfortunately, Tensorflow doesn't have very good support for multitask models.  It provides an AUC metric, but no easy way to average this metric over tasks.  We therefore must create a separate metric for every task, then define our own metric function to compute the average of them.
  
  # In[5]:
  
  
  def mean_auc(labels, predictions, weights):
      metric_ops = []
      update_ops = []
      for i in range(n_tasks):
          metric, update = tf.metrics.auc(labels[:,i], predictions[:,i], weights[:,i])
          metric_ops.append(metric)
          update_ops.append(update)
      mean_metric = tf.reduce_mean(tf.stack(metric_ops))
      update_all = tf.group(*update_ops)
      return mean_metric, update_all
  
  
  # Now we create our `Estimator` by calling `make_estimator()` on the DeepChem model.  We provide as arguments the objects created above to represent the feature and weight columns, as well as our metric function.
  
  # In[6]:
  
  
  estimator = model.make_estimator(feature_columns=[x_col],
                                   weight_column=weight_col,
                                   metrics={'mean_auc': mean_auc},
                                   model_dir='estimator')
  
  
  # We are finally ready to train and evaluate it!  Notice how the input function passed to each method is actually a lambda.  This allows us to write a single function, then use it with different datasets and numbers of epochs.
  
  # In[7]:
  
  
  estimator.train(input_fn=lambda: input_fn(train_dataset, 100))
  print(estimator.evaluate(input_fn=lambda: input_fn(test_dataset, 1)))
  
  
  # That's a lot of code for something DeepChem can do in three lines.  The Tensorflow API is verbose and somewhat confusing.  It has seemingly arbitrary limitations, like assuming a model will only ever have one output, and therefore only allowing one label.  But for better or worse, it's a standard.
  # 
  # Of course, if you just want to use a DeepChem model with a DeepChem dataset, there is no need for any of this.  Just use the DeepChem API.  But perhaps you want to use a DeepChem dataset with a model that has been implemented as an estimator.  In that case, `Dataset.make_iterator()` allows you to easily do that.  Or perhaps you have higher level workflow code that is written to work with estimators.  In that case, `make_estimator()` allows DeepChem models to easily fit into that workflow.

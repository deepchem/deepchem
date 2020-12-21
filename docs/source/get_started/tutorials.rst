Tutorials
=========

If you're new to DeepChem, you probably want to know the basics. What is DeepChem? 
Why should you care about using it? The short answer is that DeepChem is a scientific machine learning library. 
(The "Chem" indicates the historical fact that DeepChem initially focused on chemical applications,
but we aim to support all types of scientific applications more broadly).

Why would you want to use DeepChem instead of another machine learning
library? Simply put, DeepChem maintains an extensive collection of utilities
to enable scientific deep learning including classes for loading scientific
datasets, processing them, transforming them, splitting them up, and learning
from them. Behind the scenes DeepChem uses a variety of other machine
learning frameworks such as `scikit-learn`_, `TensorFlow`_, and `XGBoost`_. We are
also experimenting with adding additional models implemented in `PyTorch`_
and `JAX`_. Our focus is to facilitate scientific experimentation using
whatever tools are available at hand.

In the rest of this tutorials, we'll provide a rapid fire overview of DeepChem's API.
DeepChem is a big library so we won't cover everything, but we should give you enough to get started.

.. contents:: Contents
    :local:

Data Handling
-------------

The :code:`dc.data` module contains utilities to handle :code:`Dataset`
objects. These :code:`Dataset` objects are the heart of DeepChem.
A :code:`Dataset` is an abstraction of a dataset in machine learning. That is,
a collection of features, labels, weights, alongside associated identifiers.
Rather than explaining further, we'll just show you.

.. doctest:: 

   >>> import deepchem as dc
   >>> import numpy as np
   >>> N_samples = 50
   >>> n_features = 10
   >>> X = np.random.rand(N_samples, n_features)
   >>> y = np.random.rand(N_samples)
   >>> dataset = dc.data.NumpyDataset(X, y)
   >>> dataset.X.shape
   (50, 10)
   >>> dataset.y.shape
   (50,)

Here we've used the :code:`NumpyDataset` class which stores datasets in memory.
This works fine for smaller datasets and is very convenient for experimentation,
but is less convenient for larger datasets. For that we have the :code:`DiskDataset` class.

.. doctest::

   >>> dataset = dc.data.DiskDataset.from_numpy(X, y)
   >>> dataset.X.shape
   (50, 10)
   >>> dataset.y.shape
   (50,)

In this example we haven't specified a data directory, so this :code:`DiskDataset` is written
to a temporary folder. Note that :code:`dataset.X` and :code:`dataset.y` load data
from disk underneath the hood! So this can get very expensive for larger datasets.


Feature Engineering
-------------------

"Featurizer" is a chunk of code which transforms raw input data into a processed
form suitable for machine learning. The :code:`dc.feat` module contains an extensive collection
of featurizers for molecules, molecular complexes and inorganic crystals.
We'll show you the example about the usage of featurizers.

.. doctest::

   >>> smiles = [
   ...   'O=Cc1ccc(O)c(OC)c1',
   ...   'CN1CCC[C@H]1c2cccnc2',
   ...   'C1CCCCC1',
   ...   'c1ccccc1',
   ...   'CC(=O)O',
   ... ]
   >>> properties = [0.4, -1.5, 3.2, -0.2, 1.7]
   >>> featurizer = dc.feat.CircularFingerprint(size=1024)
   >>> ecfp = featurizer.featurize(smiles)
   >>> ecfp.shape
   (5, 1024)
   >>> dataset = dc.data.NumpyDataset(X=ecfp, y=np.array(properties))
   >>> len(dataset)
   5

Here, we've used the :code:`CircularFingerprint` and converted SMILES to ECFP.
The ECFP is a fingerprint which is a bit vector made by chemical structure information
and we can use it as the input for various models.

And then, you may have a CSV file which contains SMILES and property like HOMO-LUMO gap. 
In such a case, by using :code:`DataLoader`, you can load and featurize your data at once.

.. doctest::

   >>> import pandas as pd
   >>> # make a dataframe object for creating a CSV file
   >>> df = pd.DataFrame(list(zip(smiles, properties)), columns=["SMILES", "property"])
   >>> import tempfile
   >>> with dc.utils.UniversalNamedTemporaryFile(mode='w') as tmpfile:
   ...   # dump the CSV file
   ...   df.to_csv(tmpfile.name)
   ...   # initizalize the featurizer
   ...   featurizer = dc.feat.CircularFingerprint(size=1024)
   ...   # initizalize the dataloader
   ...   loader = dc.data.CSVLoader(["property"], feature_field="SMILES", featurizer=featurizer)
   ...   # load and featurize the data from the CSV file
   ...   dataset = loader.create_dataset(tmpfile.name)
   ...   len(dataset)
   5


Data Splitting
--------------

The :code:`dc.splits` module contains a collection of scientifically aware splitters.
Generally, we need to split the original data to training, validation and test data
in order to tune the model and evaluate the model's performance.
We'll show you the example about the usage of splitters.

.. doctest::

   >>> splitter = dc.splits.RandomSplitter()
   >>> # split 5 datapoints in the ratio of train:valid:test = 3:1:1
   >>> train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
   ...   dataset=dataset, frac_train=0.6, frac_valid=0.2, frac_test=0.2
   ... )
   >>> len(train_dataset)
   3
   >>> len(valid_dataset)
   1
   >>> len(test_dataset)
   1

Here, we've used the :code:`RandomSplitter` and splitted the data randomly
in the ratio of train:valid:test = 3:1:1. But, the random splitting sometimes
overestimates  model's performance, especially for small data or imbalance data.
Please be careful for model evaluation. The :code:`dc.splits` provides more methods
and algorithms to evaluate the model's performance appropriately, like cross validation or
splitting using molecular scaffolds.


Model Training and Evaluating
-----------------------------

The :code:`dc.models` conteins an extensive collection of models for scientific applications. 
Most of all models inherits  :code:`dc.models.Model` and we can train them by just calling :code:`fit` method.
You don't need to care about how to use specific framework APIs.
We'll show you the example about the usage of models.

.. doctest::

   >>> from sklearn.ensemble import RandomForestRegressor
   >>> rf = RandomForestRegressor()
   >>> model = dc.models.SklearnModel(model=rf)
   >>> # model training
   >>> model.fit(train_dataset)
   >>> valid_preds = model.predict(valid_dataset)
   >>> valid_preds.shape
   (1,)
   >>> test_preds = model.predict(test_dataset)
   >>> test_preds.shape
   (1,)

Here, we've used the :code:`SklearnModel` and trained the model.
Even if you want to train a deep learning model which is implemented
by TensorFlow or PyTorch, calling :code:`fit` method is all you need!

And then, if you use :code:`dc.metrics.Metric`, you can evaluate your model
by just calling :code:`evaluate` method.

.. doctest::

   >>> # initialze the metric
   >>> metric = dc.metrics.Metric(dc.metrics.mae_score)
   >>> # evaluate the model
   >>> train_score = model.evaluate(train_dataset, [metric])
   >>> valid_score = model.evaluate(valid_dataset, [metric])
   >>> test_score = model.evaluate(test_dataset, [metric])


More Tutorials
--------------

DeepChem maintains `an extensive collection of addition tutorials`_ that are meant to
be run on `Google Colab`_, an online platform that allows you to execute Jupyter notebooks.
Once you've finished this introductory tutorial, we recommend working through these more involved tutorials.

.. _`scikit-learn`: https://scikit-learn.org/stable/
.. _`TensorFlow`: https://www.tensorflow.org/
.. _`XGBoost`: https://xgboost.readthedocs.io/en/latest/
.. _`PyTorch`: https://pytorch.org/
.. _`JAX`: https://github.com/google/jax
.. _`an extensive collection of addition tutorials`: https://github.com/deepchem/deepchem/tree/master/examples/tutorials	
.. _`Google Colab`: https://colab.research.google.com/

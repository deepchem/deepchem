Examples
========

SAMPL models
============

Some examples of training models on the SAMPL(FreeSolv) dataset included in :code:`dc.molnet`.

We'll be using its ``smiles`` field to train models to predict its experimentally measured solvation energy (``expt``).

First, we'll load our libraries:

.. doctest::

   >>> import numpy as np
   >>> import tensorflow as tf
   >>> import deepchem as dc
   >>> from deepchem.molnet import load_sampl
   >>> 
   >>> # for reproducibility 
   >>> np.random.seed(123)
   >>> tf.random.set_seed(123)


.. doctest:: 

   >>> # Load SAMPL dataset
   >>> SAMPL_tasks, SAMPL_datasets, transformers = load_sampl()
   >>> SAMPL_tasks
   ['expt']
   >>> train_dataset, valid_dataset, test_dataset = SAMPL_datasets
   >>>
   >>> # We'll train a multitask regressor (fully connected network)
   >>> metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)
   >>> 
   >>> model = dc.models.MultitaskRegressor(
   ...     len(SAMPL_tasks),
   ...     n_features = 1024,
   ...     layer_sizes=[1000],
   ...     dropouts=[.25],
   ...     learning_rate=0.001,
   ...     batch_size=50)
   >>> 
   >>> # Fit trained model (returns average loss over the most recent checkpoint interval)
   >>> model.fit(train_dataset)
   0.1726440668106079
   >>> 
   >>> # We now evaluate our fitted model on our train and test sets
   >>> model.evaluate(train_dataset, [metric], transformers)
   {'mean-pearson_r2_score': 0.9244964295814636}
   >>> model.evaluate(valid_dataset, [metric], transformers)
   {'mean-pearson_r2_score': 0.7532658569385681}

For a :code:`GraphConvModel` we'll need to reload with the appropriate featurizer:
.. doctest:: 

   >>> # for reproducibility 
   >>> np.random.seed(123)
   >>> tf.random.set_seed(123)
   >>> # Load SAMPL dataset
   >>> SAMPL_tasks, SAMPL_datasets, transformers = load_sampl(
   ...     featurizer='GraphConv')
   >>> train_dataset, valid_dataset, test_dataset = SAMPL_datasets
   >>>
   >>> model = dc.models.GraphConvModel(len(SAMPL_tasks), mode='regression')
   >>> 
   >>> # Fit trained model (returns average loss over the most recent checkpoint interval)
   >>> model.fit(train_dataset, nb_epoch=20)
   0.05753047466278076
   >>> 
   >>> model.evaluate(train_dataset, [metric], transformers)
   {'mean-pearson_r2_score': 0.5772751202910659}
   >>> model.evaluate(valid_dataset, [metric], transformers)
   {'mean-pearson_r2_score': 0.36771456280565507}

Examples
========

Before jumping in to examples, we'll import our libraries and ensure our `doctests <https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html>`_ are reproducible:

.. doctest:: *

    >>> import numpy as np
    >>> import tensorflow as tf
    >>> import deepchem as dc
    >>> 
    >>> # Run before every test for reproducibility
    >>> def seed_all():
    ...     np.random.seed(123)
    ...     tf.random.set_seed(123)
    >>>

.. testsetup:: *

    import numpy as np
    import tensorflow as tf
    import deepchem as dc

    # Run before every test for reproducibility
    def seed_all():
        np.random.seed(123)
        tf.random.set_seed(123)


SAMPL (FreeSolv)
----------------

Examples of training models on the SAMPL(FreeSolv) dataset included in `MoleculeNet <./moleculenet.html>`_.

We'll be using its :code:`smiles` field to train models to predict its experimentally measured solvation energy (:code:`expt`).

MultitaskRegressor
^^^^^^^^^^^^^^^^^^

First, we'll load the dataset with :func:`load_sampl() <deepchem.molnet.load_sampl>` and fit a :class:`MultitaskRegressor <deepchem.models.MultitaskRegressor>`:

.. doctest:: sampl

    >>> seed_all()
    >>> # Load SAMPL dataset with default 'index' splitting
    >>> SAMPL_tasks, SAMPL_datasets, transformers = dc.molnet.load_sampl()
    >>> SAMPL_tasks
    ['expt']
    >>> train_dataset, valid_dataset, test_dataset = SAMPL_datasets
    >>>
    >>> # We want to know the pearson R squared score, averaged across tasks
    >>> avg_pearson_r2 = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)
    >>> 
    >>> # We'll train a multitask regressor (fully connected network)
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
    >>> # We now evaluate our fitted model on our training and validation sets
    >>> model.evaluate(train_dataset, [avg_pearson_r2], transformers)
    {'mean-pearson_r2_score': 0.9244964295814636}
    >>> model.evaluate(valid_dataset, [avg_pearson_r2], transformers)
    {'mean-pearson_r2_score': 0.7532658569385681}


GraphConvModel
^^^^^^^^^^^^^^
The default `featurizer <./featurizers.html>`_ for SAMPL is :code:`ECFP`, short for
`"Extended-connectivity fingerprints." <./featurizers.html#circularfingerprint>`_
For a :class:`GraphConvModel <deepchem.models.GraphConvModel>`, we'll reload our datasets with :code:`featurizer='GraphConv'`:

.. doctest:: sampl

    >>> seed_all()
    >>> # Load SAMPL dataset
    >>> SAMPL_tasks, SAMPL_datasets, transformers = dc.molnet.load_sampl(
    ...     featurizer='GraphConv')
    >>> train_dataset, valid_dataset, test_dataset = SAMPL_datasets
    >>>
    >>> model = dc.models.GraphConvModel(len(SAMPL_tasks), mode='regression')
    >>> 
    >>> # Fit trained model (returns average loss over the most recent checkpoint interval)
    >>> model.fit(train_dataset, nb_epoch=20)
    0.05753047466278076
    >>> 
    >>> # We now evaluate our fitted model on our training and validation sets
    >>> model.evaluate(train_dataset, [avg_pearson_r2], transformers)
    {'mean-pearson_r2_score': 0.5772751202910659}
    >>> model.evaluate(valid_dataset, [avg_pearson_r2], transformers)
    {'mean-pearson_r2_score': 0.36771456280565507}


..
  ChEMBL
  -------

  Examples of training models on `ChEMBL <https://www.ebi.ac.uk/chembl/>` dataset included in `MoleculeNet <./moleculenet.html>`_.

    ChEMBL is a manually curated database of bioactive molecules with drug-like properties.
    It brings together chemical, bioactivity and genomic data to aid the translation of genomic information into effective new drugs.

  MultitaskRegressor
  ^^^^^^^^^^^^^^^^^^

  .. doctest:: chembl

      >>> seed_all()
      >>> # Load ChEMBL 5thresh dataset with random splitting
      >>> chembl_tasks, datasets, transformers = dc.molnet.load_chembl(
      ...     shard_size=2000, featurizer="ECFP", set="5thresh", split="random")
      >>> train_dataset, valid_dataset, test_dataset = datasets
      >>> len(chembl_tasks)
      691
      >>> f'Compound train/valid/test split: {len(train_dataset)}/{len(valid_dataset)}/{len(test_dataset)}'
      'Compound train/valid/test split: 19096/2387/2388'
      >>>
      >>> # We want to know the pearson R squared score, averaged across tasks
      >>> avg_pearson_r2 = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)
      >>>
      >>> # Create our model
      >>> n_layers = 3
      >>> model = dc.models.MultitaskRegressor(
      ...     len(chembl_tasks),
      ...     train_dataset.get_data_shape()[0],
      ...     layer_sizes=[1000] * n_layers,
      ...     dropouts=[.25] * n_layers,
      ...     weight_init_stddevs=[.02] * n_layers,
      ...     bias_init_consts=[1.] * n_layers,
      ...     learning_rate=.0003,
      ...     weight_decay_penalty=.0001,
      ...     batch_size=100,
      ...     verbosity="high")
      >>>
      >>> model.fit(train_dataset, nb_epoch=10) # orig. 20
      0.04922508895397186
      >>> # We now evaluate our fitted model on our training, validation, and test sets
      >>> model.evaluate(train_dataset, [avg_pearson_r2], transformers)
      {'mean-pearson_r2_score': nan}
      >>> model.evaluate(valid_dataset, [avg_pearson_r2], transformers)
      {'mean-pearson_r2_score': nan}
      >>> model.evaluate(test_dataset, [avg_pearson_r2], transformers)
      {'mean-pearson_r2_score': nan}

  GraphConvModel
  ^^^^^^^^^^^^^^

  .. doctest:: chembl

      >>> # Load ChEMBL dataset
      >>> chembl_tasks, datasets, transformers = dc.molnet.load_chembl(
      ...    shard_size=2000, featurizer="GraphConv", set="5thresh", split="random")
      >>> train_dataset, valid_dataset, test_dataset = datasets
      >>> 
      >>> # pearson R squared score, averaged across tasks
      >>> avg_pearson_r2 = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)
      >>>
      >>> model = dc.models.GraphConvModel(
      ...    len(chembl_tasks), batch_size=128, mode='regression')
      >>>
      >>> # Fit trained model
      >>> model.fit(train_dataset, nb_epoch=20)
      None
      >>>
      >>> # We now evaluate our fitted model on our training, validation, and test sets
      >>> model.evaluate(train_dataset, [avg_pearson_r2], transformers)
      {'mean-pearson_r2_score': nan}
      >>> model.evaluate(valid_dataset, [avg_pearson_r2], transformers) and False
      {'mean-pearson_r2_score': nan}
      >>> model.evaluate(test_dataset, [avg_pearson_r2], transformers) and False
      {'mean-pearson_r2_score': nan}


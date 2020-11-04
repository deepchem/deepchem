Examples
========

We show a bunch of examples for DeepChem by the doctest style.

- We match against doctest's :code:`...` wildcard on code where output is usually ignored
- We often use threshold assertions (e.g: :code:`score['mean-pearson_r2_score'] > 0.92`),
  as this is what matters for model training code.

.. contents:: Contents
    :local:

Before jumping in to examples, we'll import our libraries and ensure our doctests are reproducible:

.. doctest:: *

    >>> import numpy as np
    >>> import tensorflow as tf
    >>> import deepchem as dc
    >>>
    >>> # Run before every test for reproducibility
    >>> def seed_all():
    ...     np.random.seed(123)
    ...     tf.random.set_seed(123)

.. testsetup:: *

    import numpy as np
    import tensorflow as tf
    import deepchem as dc

    # Run before every test for reproducibility
    def seed_all():
        np.random.seed(123)
        tf.random.set_seed(123)


Delaney (ESOL)
----------------

Examples of training models on the Delaney (ESOL) dataset included in `MoleculeNet <./moleculenet.html>`_.

We'll be using its :code:`smiles` field to train models to predict its experimentally measured solvation energy (:code:`expt`).

MultitaskRegressor
^^^^^^^^^^^^^^^^^^

First, we'll load the dataset with :func:`load_delaney() <deepchem.molnet.load_delaney>` and fit a :class:`MultitaskRegressor <deepchem.models.MultitaskRegressor>`:

.. doctest:: delaney

    >>> seed_all()
    >>> # Load dataset with default 'scaffold' splitting
    >>> tasks, datasets, transformers = dc.molnet.load_delaney()
    >>> tasks
    ['measured log solubility in mols per litre']
    >>> train_dataset, valid_dataset, test_dataset = datasets
    >>>
    >>> # We want to know the pearson R squared score, averaged across tasks
    >>> avg_pearson_r2 = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)
    >>>
    >>> # We'll train a multitask regressor (fully connected network)
    >>> model = dc.models.MultitaskRegressor(
    ...     len(tasks),
    ...     n_features=1024,
    ...     layer_sizes=[500])
    >>>
    >>> model.fit(train_dataset)
    0...
    >>>
    >>> # We now evaluate our fitted model on our training and validation sets
    >>> train_scores = model.evaluate(train_dataset, [avg_pearson_r2], transformers)
    >>> assert train_scores['mean-pearson_r2_score'] > 0.7, train_scores
    >>>
    >>> valid_scores = model.evaluate(valid_dataset, [avg_pearson_r2], transformers)
    >>> assert valid_scores['mean-pearson_r2_score'] > 0.3, valid_scores


GraphConvModel
^^^^^^^^^^^^^^
The default `featurizer <./featurizers.html>`_ for Delaney is :code:`ECFP`, short for
`"Extended-connectivity fingerprints." <./featurizers.html#circularfingerprint>`_
For a :class:`GraphConvModel <deepchem.models.GraphConvModel>`, we'll reload our datasets with :code:`featurizer='GraphConv'`:

.. doctest:: delaney

    >>> seed_all()
    >>> tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='GraphConv')
    >>> train_dataset, valid_dataset, test_dataset = datasets
    >>>
    >>> model = dc.models.GraphConvModel(len(tasks), mode='regression', dropout=0.5)
    >>>
    >>> model.fit(train_dataset, nb_epoch=30)
    0...
    >>>
    >>> # We now evaluate our fitted model on our training and validation sets
    >>> train_scores = model.evaluate(train_dataset, [avg_pearson_r2], transformers)
    >>> assert train_scores['mean-pearson_r2_score'] > 0.5, train_scores
    >>>
    >>> valid_scores = model.evaluate(valid_dataset, [avg_pearson_r2], transformers)
    >>> assert valid_scores['mean-pearson_r2_score'] > 0.3, valid_scores


ChEMBL
------

Examples of training models on `ChEMBL`_ dataset included in MoleculeNet.

ChEMBL is a manually curated database of bioactive molecules with drug-like properties.
It brings together chemical, bioactivity and genomic data to aid the translation
of genomic information into effective new drugs.

.. _`ChEMBL`: https://www.ebi.ac.uk/chembl

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
    >>> # We want to know the RMS, averaged across tasks
    >>> avg_rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean)
    >>>
    >>> # Create our model
    >>> n_layers = 3
    >>> model = dc.models.MultitaskRegressor(
    ...     len(chembl_tasks),
    ...     n_features=1024,
    ...     layer_sizes=[1000] * n_layers,
    ...     dropouts=[.25] * n_layers,
    ...     weight_init_stddevs=[.02] * n_layers,
    ...     bias_init_consts=[1.] * n_layers,
    ...     learning_rate=.0003,
    ...     weight_decay_penalty=.0001,
    ...     batch_size=100)
    >>>
    >>> model.fit(train_dataset, nb_epoch=5)
    0...
    >>>
    >>> # We now evaluate our fitted model on our training and validation sets
    >>> train_scores = model.evaluate(train_dataset, [avg_rms], transformers)
    >>> assert train_scores['mean-rms_score'] < 10.00
    >>>
    >>> valid_scores = model.evaluate(valid_dataset, [avg_rms], transformers)
    >>> assert valid_scores['mean-rms_score'] < 10.00

GraphConvModel
^^^^^^^^^^^^^^

.. doctest:: chembl

    >>> # Load ChEMBL dataset
    >>> chembl_tasks, datasets, transformers = dc.molnet.load_chembl(
    ...    shard_size=2000, featurizer="GraphConv", set="5thresh", split="random")
    >>> train_dataset, valid_dataset, test_dataset = datasets
    >>>
    >>> # RMS, averaged across tasks
    >>> avg_rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean)
    >>>
    >>> model = dc.models.GraphConvModel(
    ...    len(chembl_tasks), batch_size=128, mode='regression')
    >>>
    >>> # Fit trained model
    >>> model.fit(train_dataset, nb_epoch=5)
    0...
    >>>
    >>> # We now evaluate our fitted model on our training and validation sets
    >>> train_scores = model.evaluate(train_dataset, [avg_rms], transformers)
    >>> assert train_scores['mean-rms_score'] < 10.00
    >>>
    >>> valid_scores = model.evaluate(valid_dataset, [avg_rms], transformers)
    >>> assert valid_scores['mean-rms_score'] < 10.00

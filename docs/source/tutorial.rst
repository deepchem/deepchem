DeepChem Tutorial
=================

If you're new to DeepChem, you probably want to know the basics. What is DeepChem? Why should you care about using it? The short answer is that DeepChem is a scientific machine learning library. (The "Chem" indicates the historical fact that DeepChem initially focused on chemical applications, but we aim to support all types of scientific applications more broadly).

Why would you want to use DeepChem instead of another machine learning
library? Simply put, DeepChem maintains an extensive collection of utilities
to enable scientific deep learning including classes for loading scientific
datasets, processing them, transforming them, splitting them up, and learning
from them. Behind the scenes DeepChem uses a variety of other machine
learning frameworks such as `sklearn`_, `tensorflow`_, and `xgboost`_. We are
also experimenting with adding additional models implemented in `pytorch`_
and `jax`_. Our focus is to facilitate scientific experimentation using
whatever tools are available at hand.

In the rest of this tutorials, we'll provide a rapid fire overview of DeepChem's API. DeepChem is a big library so we won't cover everything, but we should give you enough to get started.

.. _`sklearn`: https://scikit-learn.org/stable/

.. _`tensorflow`: https://www.tensorflow.org/

.. _`xgboost`: https://xgboost.readthedocs.io/en/latest/

.. _`pytorch`: https://pytorch.org/

.. _`jax`: https://github.com/google/jax


Quickstart
----------
If you're new, you can install DeepChem on a new machine with the following commands

.. code-block:: bash

    pip install tensorflow==2.2.0
    pip install --pre deepchem


DeepChem is under very active development at present, so we recommend using our nightly build until we release a next major release. Note that to use DeepChem for chemistry applications, you will have to also install RDKit using conda.

.. code-block:: bash

    conda install -y -c conda-forge rdkit



Datasets
--------
The :code:`dc.data` module contains utilities to handle :code:`Dataset`
objects. These :code:`Dataset` objects are the heart of DeepChem. A
:code:`Dataset` is an abstraction of a dataset in machine learning. That is,
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

Here we've used the :code:`NumpyDataset` class which stores datasets in memory. This works fine for smaller datasets and is very convenient for experimentation, but is less convenient for larger datasets. For that we have the :code:`DiskDataset` class.

.. doctest::

   >>> dataset = dc.data.DiskDataset.from_numpy(X, y)
   >>> dataset.X.shape
   (50, 10)
   >>> dataset.y.shape
   (50,)

In this example we haven't specified a data directory, so this :code:`DiskDataset` is written to a temporary folder. Note that :code:`dataset.X` and :code:`dataset.y` load data from disk underneath the hood! So this can get very expensive for larger datasets.


More Tutorials
--------------
DeepChem maintains an extensive collection of addition `tutorials`_ that are meant to be run on Google `colab`_, an online platform that allows you to execute Jupyter notebooks. Once you've finished this introductory tutorial, we recommend working through these more involved tutorials.

.. _`tutorials`: https://github.com/deepchem/deepchem/tree/master/examples/tutorials

.. _`colab`: https://colab.research.google.com/

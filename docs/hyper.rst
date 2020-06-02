Hyperparameter Tuning
=====================
One of the most important aspects of machine learning is
hyperparameter tuning. Many machine learning models have a number of
hyperparameters that control aspects of the model. These
hyperparameters typically cannot be learned directly by the same
learning algorithm used for the rest of learning and have to be set in
an alternate fashion. The :code:`dc.hyper` module contains utilities
for hyperparameter tuning.

Hyperparameter Optimization API
-------------------------------

.. autoclass:: deepchem.hyper.HyperparamOpt
  :members:
  :special-members:

Grid Hyperparameter Optimization
--------------------------------

This is the simplest form of hyperparameter optimization that simply
involves iterating over a fixed grid of possible values for
hyperaparameters.

.. autoclass:: deepchem.hyper.GridHyperparamOpt
  :members:
  :special-members:

Gaussian Process Hyperparameter Optimization
--------------------------------------------

.. autoclass:: deepchem.hyper.GaussianProcessHyperparamOpt
  :members:
  :special-members:



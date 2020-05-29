Metrics
=======
Metrics are one of the most import parts of machine learning. Unlike
traditional software, in which algorithms either work or don't work,
machine learning models work in degrees. That is, there's a continuous
range of "goodness" for a model. "Metrics" are functions which measure
how well a model works. There are many different choices of metrics
depending on the type of model at hand.

Metric Utilities
----------------

.. autofunction:: deepchem.metrics.to_one_hot

.. autofunction:: deepchem.metrics.from_one_hot


Metric Functions
----------------
.. autofunction:: deepchem.metrics.roc_auc_score

.. autofunction:: deepchem.metrics.accuracy_score

.. autofunction:: deepchem.metrics.balanced_accuracy_score

.. autofunction:: deepchem.metrics.pearson_r2_score

.. autofunction:: deepchem.metrics.jaccard_index

.. autofunction:: deepchem.metrics.pixel_error

.. autofunction:: deepchem.metrics.prc_auc_score

.. autofunction:: deepchem.metrics.rms_score

.. autofunction:: deepchem.metrics.mae_score

.. autofunction:: deepchem.metrics.kappa_score

.. autofunction:: deepchem.metrics.bedroc_score

.. autofunction:: deepchem.metrics.genomic_metrics.get_motif_scores

.. autofunction:: deepchem.metrics.genomic_metrics.get_pssm_scores

.. autofunction:: deepchem.metrics.genomic_metrics.in_silico_mutagenesis

Metric Class
------------
The :code:`dc.metrics.Metric` class is a wrapper around metric
functions which interoperates with DeepChem :code:`dc.models.Model`.

.. autoclass:: deepchem.metrics.Metric
  :members:

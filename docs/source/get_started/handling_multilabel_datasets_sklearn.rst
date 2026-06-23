.. _multilabel-missing-labels:

Handling Multilabel Datasets with Missing Labels
===========================================================

DeepChem supports multilabel classification using Scikit-learn estimators. However, many Scikit-learn models do not handle missing labels (NaN). To address this, DeepChem splits the dataset into separate single-task datasets and trains independent models for each label.

This tutorial explains how to:

- Handle multilabel datasets with missing values.
- Train separate models for each task.
- Evaluate performance using per-task and average metrics.

DeepChem Approach
-------------------

**Dataset Preparation**

- A multilabel dataset is loaded, and missing labels (NaN) are identified.
- The dataset is split into multiple single-task datasets, each corresponding to a label.

**Training Process**

- Each label (task) is treated as a separate binary classification problem.
- Models like `RandomForestClassifier` (or other Scikit-learn models) are trained separately for each task.

**Prediction & Evaluation**

- Each model predicts its respective task.
- Predictions are concatenated to form the final multilabel output.
- Metrics (e.g., accuracy, F1-score) are computed separately for each task and then averaged.

Setting Up the Environment
--------------------------
Ensure you have DeepChem installed in your Python environment. You can install it using:

.. code-block:: bash

   pip install deepchem

Additionally, install Scikit-learn if not already available:

.. code-block:: bash

   pip install scikit-learn

Working with Multilabel Datasets
--------------------------------

Step 1: Load a Multilabel Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DeepChem provides utilities to load datasets. Here, we demonstrate loading a sample multilabel dataset:

.. code-block:: python

   import deepchem as dc
   import numpy as np
   from deepchem.molnet import load_tox21

   # Load Tox21 dataset (multilabel classification)
   tasks, datasets, transformers = load_tox21(featurizer='ECFP')
   train_dataset, valid_dataset, test_dataset = datasets

   # Convert dataset to NumPy for easier manipulation
   X_train, y_train, w_train = train_dataset.X, train_dataset.y, train_dataset.w

Step 2: Handle Missing Labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since many Scikit-learn models do not support NaN values in labels, we need to process them:

.. code-block:: python

   def split_into_single_task_datasets(X, y, tasks):
       task_datasets = []
       for task_idx in range(y.shape[1]):
           mask = ~np.isnan(y[:, task_idx])  # Remove NaN values
           task_X, task_y = X[mask], y[mask, task_idx]
           task_datasets.append((task_X, task_y))
       return task_datasets

   single_task_datasets = split_into_single_task_datasets(X_train, y_train, tasks)

Step 3: Train Separate Models for Each Task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each task will have its own classifier:

.. code-block:: python

   from sklearn.ensemble import RandomForestClassifier

   models = []
   for task_X, task_y in single_task_datasets:
       model = RandomForestClassifier(n_estimators=100)
       model.fit(task_X, task_y)
       models.append(model)

Step 4: Predict on New Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To predict for new samples, we concatenate predictions from all task models:

.. code-block:: python

   def predict_multilabel(models, X):
       predictions = [model.predict(X) for model in models]
       return np.column_stack(predictions)

   multilabel_predictions = predict_multilabel(models, X_train)

Step 5: Evaluate Model Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Metrics such as accuracy and F1-score can be computed separately for each task and averaged:

.. code-block:: python

   from sklearn.metrics import accuracy_score, f1_score

   def evaluate_multilabel(y_true, y_pred):
       accuracies = [accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
       f1_scores = [f1_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
       return np.mean(accuracies), np.mean(f1_scores)

   avg_accuracy, avg_f1 = evaluate_multilabel(y_train, multilabel_predictions)
   print(f'Average Accuracy: {avg_accuracy:.2f}, Average F1-score: {avg_f1:.2f}')

Conclusion
----------
This tutorial demonstrated how DeepChem processes multilabel datasets with missing labels by:

- **Splitting** datasets into single-task versions.
- **Training** separate models for each task.
- **Predicting and evaluating** performance efficiently.


This method ensures compatibility with Scikit-learn estimators and maintains robust model performance across multiple labels.


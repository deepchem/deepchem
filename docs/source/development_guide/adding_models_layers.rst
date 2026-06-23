Contributing Models and Layers
=============================

This guide provides a comprehensive overview of how to add new machine learning models and layers to DeepChem. DeepChem is designed to be extensible, allowing researchers and developers to easily integrate their own architectures.

When to Add a Model vs. a Layer
-------------------------------

*   **Layer**: Add a layer if you are implementing a specific mathematical operation or a modular building block that can be reused across different models (e.g., a new graph convolution operation, an attention mechanism).
*   **Model**: Add a model if you are implementing a complete architecture that takes a dataset as input and performs a specific task like regression, classification, or generation.

Adding a New Model
------------------

DeepChem's models are typically implemented in PyTorch and inherit from ``deepchem.models.torch_models.torch_model.TorchModel``.

Step-by-step Guide
^^^^^^^^^^^^^^^^^^

1.  **Define the Base Module**: Create a class inheriting from ``nn.Module``. This class should define the architecture of your model.

    .. code-block:: python

        import torch.nn as nn

        class MyCustomModule(nn.Module):
            def __init__(self, d_input, d_output):
                super(MyCustomModule, self).__init__()
                self.layer = nn.Linear(d_input, d_output)

            def forward(self, x):
                return self.layer(x)

2.  **Create the TorchModel Wrapper**: Create a class inheriting from ``TorchModel``. This class acts as a wrapper that handles training, saving, loading, and prediction logic.

    .. code-block:: python

        from deepchem.models.torch_models.torch_model import TorchModel
        from deepchem.models.losses import L2Loss
        import torch.nn as nn

        class MyCustomModule(nn.Module):
            def __init__(self, d_input, d_output):
                super(MyCustomModule, self).__init__()
                self.layer = nn.Linear(d_input, d_output)

            def forward(self, x):
                return self.layer(x)

        class MyCustomModel(TorchModel):
            def __init__(self, d_input, d_output, **kwargs):
                model = MyCustomModule(d_input, d_output)
                loss = L2Loss()
                super(MyCustomModel, self).__init__(model, loss, **kwargs)

3.  **Placement**: Save your model file in ``deepchem/models/torch_models/``. For example, ``deepchem/models/torch_models/my_custom_model.py``.

4.  **Registration**: Add your model to ``deepchem/models/__init__.py`` and ``deepchem/models/torch_models/__init__.py`` to make it accessible via ``dc.models.MyCustomModel``.

Handling Different Modes
^^^^^^^^^^^^^^^^^^^^^^^^

If your model supports both classification and regression, you should handle them in the ``__init__`` and potentially in ``forward``.

.. code-block:: python

    from deepchem.models.losses import L2Loss, SparseSoftmaxCrossEntropy

    def __init__(self, n_tasks, mode='regression', n_classes=2, **kwargs):
        self.mode = mode
        if mode == 'classification':
            out_size = n_tasks * n_classes
            loss = SparseSoftmaxCrossEntropy()
        else:
            out_size = n_tasks
            loss = L2Loss()
        # ... initialize module and super ...

Adding a New Layer
------------------

Layers should be modular and reusable. Most layers in DeepChem are implemented in ``deepchem/models/torch_models/layers.py``.

Example: A Reusable MLP Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch.nn as nn

    class MultilayerPerceptron(nn.Module):
        def __init__(self, d_input, d_output, d_hidden=(64, 64)):
            super(MultilayerPerceptron, self).__init__()
            layers = []
            in_dim = d_input
            for h in d_hidden:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            layers.append(nn.Linear(in_dim, d_output))
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

Writing Unit Tests
------------------

Tests are essential for any new contribution. Your tests should live in ``deepchem/models/torch_models/tests/``.

Using Pytest Markers
^^^^^^^^^^^^^^^^^^^^

DeepChem uses markers to categorize tests. For PyTorch models, use:

.. code-block:: python

    import pytest

    @pytest.mark.torch
    def test_my_model():
        # ... test code ...

A Complete Test Example
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import deepchem as dc
    import numpy as np
    import pytest

    @pytest.mark.torch
    def test_overfit_custom_model():
        n_tasks = 1
        n_samples = 10
        n_features = 3
        np.random.seed(42)
        X = np.random.rand(n_samples, n_features)
        y = np.sum(X, axis=1, keepdims=True)  # Deterministic labels
        dataset = dc.data.NumpyDataset(X, y)

        model = MyCustomModel(d_input=n_features, d_output=n_tasks)
        model.fit(dataset, nb_epoch=1000)
        scores = model.evaluate(dataset, [dc.metrics.Metric(dc.metrics.pearson_r2_score)])
        assert scores['pearson_r2_score'] > 0.9

Reloading Test Example
^^^^^^^^^^^^^^^^^^^^^^

It is also important to ensure that the model can be correctly saved and restored.

.. code-block:: python

    @pytest.mark.torch
    def test_custom_model_reload():
        n_tasks = 1
        n_features = 3
        model = MyCustomModel(d_input=n_features, d_output=n_tasks)
        model.save_checkpoint()

        model_reloaded = MyCustomModel(d_input=n_features, d_output=n_tasks,
                                        model_dir=model.model_dir)
        model_reloaded.restore()

        # Check that predictions match
        X = np.random.rand(5, n_features)
        y1 = model.predict_on_batch(X)
        y2 = model_reloaded.predict_on_batch(X)
        assert np.allclose(y1, y2)

Summary Checklist
-----------------

*   [ ] Does the code follow the :doc:`coding` conventions?
*   [ ] Are there comprehensive docstrings?
*   [ ] Are all new classes and functions type-annotated?
*   [ ] Have you added overfitting and reloading tests?
*   [ ] Does the code pass ``yapf -i <file>``?
*   [ ] Does the code pass ``flake8 <file> --count``?
*   [ ] Does the code pass ``mypy -p deepchem``?

Adding new models and layers
==================

This guide explains how to contribute a new model or layer to DeepChem. DeepChem largely supports PyTorch, so this guide focuses on implementing a ``TorchModel``.

1. Adding a New Layer
----------------------
DeepChem layers are essentially pytorch ``nn.modules`` classes

* **Location:** Add your layer file in ``deepchem/models/layers/``.
* **Structure:**
    * Inherit from ``torch.nn.Module``.
    * Implement ``__init__`` to define parameters.
    * Implement ``forward`` to define the computation.

.. code-block:: python

    import torch.nn as nn

    class CustomLayer(nn.Module):
        def __init__(self, output_dim):
            super(CustomLayer, self).__init__()
            self.dense = nn.Linear(10, output_dim)

        def forward(self, x):
            return self.dense(x)


1. Adding a New Model
----------------------
To add a custom Pytorch model into DeepChem, you can wrap it using ``TorchModel``.

* **Location:** Add your model file in ``deepchem/models/torch_models/``.
* **Steps:**
    1.  Define your PyTorch architecture (the ``nn.Module``).
    2.  Create a class that inherits from ``deepchem.models.TorchModel``.
    3.  In the ``__init__``, pass your architecture to the parent class.

.. code-block:: python

    from deepchem.models import TorchModel
    from deepchem.models.losses import L2Loss

    class NewModel(TorchModel):
        def __init__(self, **kwargs):
            model = CustomLayer(output_dim=1)
            loss = L2Loss()
            super(NewModel, self).__init__(model, loss, **kwargs)


3. Writing Unit Tests
---------------------
Every new model requires a unit test to ensure it works and can "overfit" (learn) a small dataset.

* **Location:** ``deepchem/models/tests/test_my_model.py``.
* **The "Overfit" Test:** This is the standard DeepChem test. Create a tiny, fake dataset and ensure the model error drops to near zero.

.. code-block:: python

    import deepchem as dc
    import numpy as np
    import pytest

    @pytest.mark.torch
    def test_model_overfit():
        # 1. Create fake data
        X = np.random.rand(10, 10)
        y = np.random.rand(10, 1)
        dataset = dc.data.NumpyDataset(X, y)

        # 2. Initialize model
        model = NewModel(learning_rate=0.01)

        # 3. Fit model
        model.fit(dataset, nb_epoch=100)

        # 4. Check error is low
        loss = model.evaluate(dataset, [dc.metrics.MeanSquaredError()])
        assert loss['mean_squared_error'] < 0.1


"""Test that MAML models can be reloaded."""

import deepchem as dc
import numpy as np
import pytest

try:
    from deepchem.metalearning.torch_maml import MAML
    import torch
    import torch.nn.functional as F

    class SineLearner(dc.metalearning.MetaLearner):

        def __init__(self):
            self.batch_size = 10
            self.w1 = torch.nn.Parameter(
                torch.tensor(np.random.normal(size=[1, 40], scale=1.0),
                             requires_grad=True))
            self.w2 = torch.nn.Parameter(
                torch.tensor(np.random.normal(size=[40, 40],
                                              scale=np.sqrt(1 / 40)),
                             requires_grad=True))
            self.w3 = torch.nn.Parameter(
                torch.tensor(np.random.normal(size=[40, 1],
                                              scale=np.sqrt(1 / 40)),
                             requires_grad=True))
            self.b1 = torch.nn.Parameter(torch.tensor(np.zeros(40)),
                                         requires_grad=True)
            self.b2 = torch.nn.Parameter(torch.tensor(np.zeros(40)),
                                         requires_grad=True)
            self.b3 = torch.nn.Parameter(torch.tensor(np.zeros(1)),
                                         requires_grad=True)

        def compute_model(self, inputs, variables, training):
            x, y = inputs
            w1, w2, w3, b1, b2, b3 = variables
            dense1 = F.relu(torch.matmul(x, w1) + b1)
            dense2 = F.relu(torch.matmul(dense1, w2) + b2)
            output = torch.matmul(dense2, w3) + b3
            loss = torch.mean(torch.square(output - y))
            return loss, [output]

        @property
        def variables(self):
            return [self.w1, self.w2, self.w3, self.b1, self.b2, self.b3]

        def select_task(self):
            self.amplitude = 5.0 * np.random.random()
            self.phase = np.pi * np.random.random()

        def get_batch(self):
            x = torch.tensor(np.random.uniform(-5.0, 5.0, (self.batch_size, 1)))
            return [x, torch.tensor(self.amplitude * np.sin(x + self.phase))]

        def parameters(self):
            for key, value in self.__dict__.items():
                if isinstance(value, torch.nn.Parameter):
                    yield value

    has_pytorch = True
except:
    has_pytorch = False


@pytest.mark.torch
def test_reload():
    """Test that a Metalearner can be reloaded."""
    learner = SineLearner()
    optimizer = dc.models.optimizers.Adam(learning_rate=5e-3)
    maml = MAML(learner, meta_batch_size=4, optimizer=optimizer)
    maml.fit(900)

    learner.select_task()
    batch = learner.get_batch()
    loss, outputs = maml.predict_on_batch(batch)
    loss = loss.detach().numpy()

    reloaded = MAML(SineLearner(), model_dir=maml.model_dir)
    reloaded.restore()
    reloaded_loss, reloaded_outputs = maml.predict_on_batch(batch)
    reloaded_loss = reloaded_loss.detach().numpy()

    assert loss == reloaded_loss

    assert len(outputs) == len(reloaded_outputs)

    outputs = outputs[0].detach().numpy()
    reloaded_outputs = reloaded_outputs[0].detach().numpy()
    for output, reloaded_output in zip(outputs, reloaded_outputs):
        assert np.all(output == reloaded_output)

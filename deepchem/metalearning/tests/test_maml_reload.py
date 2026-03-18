"""Test that MAML models can be reloaded."""

import deepchem as dc
import numpy as np
import pytest

try:
    import tensorflow as tf

    class SineLearner(dc.metalearning.MetaLearner):

        def __init__(self):
            self.batch_size = 10
            self.w1 = tf.Variable(np.random.normal(size=[1, 40], scale=1.0))
            self.w2 = tf.Variable(
                np.random.normal(size=[40, 40], scale=np.sqrt(1 / 40)))
            self.w3 = tf.Variable(
                np.random.normal(size=[40, 1], scale=np.sqrt(1 / 40)))
            self.b1 = tf.Variable(np.zeros(40))
            self.b2 = tf.Variable(np.zeros(40))
            self.b3 = tf.Variable(np.zeros(1))

        def compute_model(self, inputs, variables, training):
            x, y = inputs
            w1, w2, w3, b1, b2, b3 = variables
            dense1 = tf.nn.relu(tf.matmul(x, w1) + b1)
            dense2 = tf.nn.relu(tf.matmul(dense1, w2) + b2)
            output = tf.matmul(dense2, w3) + b3
            loss = tf.reduce_mean(tf.square(output - y))
            return loss, [output]

        @property
        def variables(self):
            return [self.w1, self.w2, self.w3, self.b1, self.b2, self.b3]

        def select_task(self):
            self.amplitude = 5.0 * np.random.random()
            self.phase = np.pi * np.random.random()

        def get_batch(self):
            x = np.random.uniform(-5.0, 5.0, (self.batch_size, 1))
            return [x, self.amplitude * np.sin(x + self.phase)]

    has_tensorflow = True

except:
    has_tensorflow = False


@pytest.mark.tensorflow
def test_reload():
    """Test that a Metalearner can be reloaded."""
    learner = SineLearner()
    optimizer = dc.models.optimizers.Adam(learning_rate=5e-3)
    maml = dc.metalearning.MAML(learner, meta_batch_size=4, optimizer=optimizer)
    maml.fit(900)

    learner.select_task()
    batch = learner.get_batch()
    loss, outputs = maml.predict_on_batch(batch)

    reloaded = dc.metalearning.MAML(SineLearner(), model_dir=maml.model_dir)
    reloaded.restore()
    reloaded_loss, reloaded_outputs = maml.predict_on_batch(batch)

    assert loss == reloaded_loss

    assert len(outputs) == len(reloaded_outputs)
    for output, reloaded_output in zip(outputs, reloaded_outputs):
        assert np.all(output == reloaded_output)

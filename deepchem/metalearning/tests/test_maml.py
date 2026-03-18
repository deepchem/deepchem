import unittest
import pytest
import numpy as np
from flaky import flaky
import deepchem as dc

try:
    import tensorflow as tf
    has_tensorflow = True
except:
    has_tensorflow = False


class TestMAML(unittest.TestCase):

    @flaky
    @pytest.mark.tensorflow
    def test_sine(self):
        """Test meta-learning for sine function."""

        # This is a MetaLearner that learns to generate sine functions with variable
        # amplitude and phase.

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

        # Optimize it.

        learner = SineLearner()
        optimizer = dc.models.optimizers.Adam(learning_rate=5e-3)
        maml = dc.metalearning.MAML(learner,
                                    meta_batch_size=4,
                                    optimizer=optimizer)
        maml.fit(9000)

        # Test it out on some new tasks and see how it works.

        loss1 = []
        loss2 = []
        for i in range(50):
            learner.select_task()
            maml.restore()
            batch = learner.get_batch()
            loss, outputs = maml.predict_on_batch(batch)
            loss1.append(np.sqrt(loss))
            maml.train_on_current_task()
            loss, outputs = maml.predict_on_batch(batch)
            loss2.append(np.sqrt(loss))

        # Initially the model should do a bad job of fitting the sine function.

        assert np.average(loss1) > 1.0

        # After one step of optimization it should do much better.

        assert np.average(loss2) < np.average(loss1)

        # Verify that we can create a new MAML object, reload the parameters from the first one, and
        # get the same result.

        new_maml = dc.metalearning.MAML(SineLearner(), model_dir=maml.model_dir)
        new_maml.restore()
        loss, outputs = new_maml.predict_on_batch(batch)
        assert np.sqrt(loss) == loss1[-1]

        # Do the same thing, only using the "restore" argument to fit().

        new_maml = dc.metalearning.MAML(SineLearner(), model_dir=maml.model_dir)
        new_maml.fit(0, restore=True)
        loss, outputs = new_maml.predict_on_batch(batch)
        assert np.sqrt(loss) == loss1[-1]

from flaky import flaky

import deepchem as dc
from deepchem.models.tensorgraph.layers import Feature, Label, Dense, L2Loss
import numpy as np
import tensorflow as tf
import unittest


class TestMAML(unittest.TestCase):

  @flaky
  def test_sine(self):
    """Test meta-learning for sine function."""

    # This is a MetaLearner that learns to generate sine functions with variable
    # amplitude and phase.

    class SineLearner(dc.metalearning.MetaLearner):

      def __init__(self):
        self.batch_size = 10
        self.tg = dc.models.TensorGraph(use_queue=False)
        self.features = Feature(shape=(None, 1))
        self.labels = Label(shape=(None, 1))
        hidden1 = Dense(
            in_layers=self.features, out_channels=40, activation_fn=tf.nn.relu)
        hidden2 = Dense(
            in_layers=hidden1, out_channels=40, activation_fn=tf.nn.relu)
        output = Dense(in_layers=hidden2, out_channels=1)
        loss = L2Loss(in_layers=[output, self.labels])
        self.tg.add_output(output)
        self.tg.set_loss(loss)
        with self.tg._get_tf("Graph").as_default():
          self.tg.build()

      @property
      def loss(self):
        return self.tg.loss

      def select_task(self):
        self.amplitude = 5.0 * np.random.random()
        self.phase = np.pi * np.random.random()

      def get_batch(self):
        x = np.random.uniform(-5.0, 5.0, (self.batch_size, 1))
        feed_dict = {}
        feed_dict[self.features.out_tensor] = x
        feed_dict[self.labels.out_tensor] = self.amplitude * np.sin(
            x + self.phase)
        return feed_dict

    # Optimize it.

    learner = SineLearner()
    maml = dc.metalearning.MAML(learner, meta_batch_size=4)
    maml.fit(12000)

    # Test it out on some new tasks and see how it works.

    loss1 = []
    loss2 = []
    for i in range(50):
      learner.select_task()
      feed_dict = learner.get_batch()
      for key, value in learner.get_batch().items():
        feed_dict[maml._meta_placeholders[key]] = value
      loss1.append(
          np.average(
              np.sqrt(maml._session.run(maml._loss, feed_dict=feed_dict))))
      loss2.append(
          np.average(
              np.sqrt(maml._session.run(maml._meta_loss, feed_dict=feed_dict))))

    # Initially the model should do a bad job of fitting the sine function.

    assert np.average(loss1) > 1.0

    # After one step of optimization it should do much better.

    assert np.average(loss2) < 1.0

    # Verify that we can create a new MAML object, reload the parameters from the first one, and
    # get the same result.

    new_maml = dc.metalearning.MAML(learner, model_dir=maml.model_dir)
    new_maml.restore()
    new_loss = np.average(
        np.sqrt(new_maml._session.run(new_maml._loss, feed_dict=feed_dict)))
    assert new_loss == loss1[-1]

    # Do the same thing, only using the "restore" argument to fit().

    new_maml = dc.metalearning.MAML(learner, model_dir=maml.model_dir)
    new_maml.fit(0, restore=True)
    new_loss = np.average(
        np.sqrt(new_maml._session.run(new_maml._loss, feed_dict=feed_dict)))
    assert new_loss == loss1[-1]

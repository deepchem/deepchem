import deepchem as dc
import numpy as np
from tensorflow.python.framework import test_util


class TestOntology(test_util.TensorFlowTestCase):
  """Test OntologyModel."""

  def test_ontology_regressor(self):
    """Test training an OntologyModel for regression."""

    # Create a dataset with three tasks.  The first two tasks each depend only
    # on half the features.  The third task depends on all of them.

    n_features = 8
    n_tasks = 3
    n_samples = 30
    x = np.random.random((n_samples, n_features))
    y = np.zeros((n_samples, n_tasks))
    y[:, 0] = np.sum(x[:, :(n_features//2)], axis=1)
    y[:, 1] = np.sum(x[:, (n_features//2):], axis=1)
    y[:, 2] = 0.5*np.sum(x, axis=1)
    dataset = dc.data.NumpyDataset(x, y)

    # Create an OntologyModel.  Two leaf nodes contain half the features.

    feature_ids = [str(i) for i in range(n_features)]
    node1 = dc.models.OntologyNode('leaf1', 5, feature_ids[:(n_features//2)])
    node2 = dc.models.OntologyNode('leaf2', 5, feature_ids[(n_features//2):])
    node3 = dc.models.OntologyNode('root', 5, children=[node1, node2])
    model = dc.models.OntologyModel(n_tasks, feature_ids, node3, learning_rate=0.02)

    # Train the model on the datase.

    model.fit(dataset, nb_epoch=1000)

    # It should have learned to predict all of the tasks accurately.

    pred_error = np.mean((model.predict_on_batch(x)-y)**2, axis=0)
    assert np.all(pred_error < 0.01)

    # In addition, it should be able to predict the first task based only on the
    # first leaf node, and the second task based only on the second leaf node.

    leaf1_error = np.mean((model.predict_on_batch(x, outputs=model.prediction_for_node['leaf1'])-y)**2, axis=0)
    leaf2_error = np.mean((model.predict_on_batch(x, outputs=model.prediction_for_node['leaf2'])-y)**2, axis=0)
    assert leaf1_error[0] < 0.01
    assert leaf2_error[1] < 0.01
    assert np.mean([leaf1_error[1], leaf1_error[2], leaf2_error[0], leaf2_error[2]]) > 0.01

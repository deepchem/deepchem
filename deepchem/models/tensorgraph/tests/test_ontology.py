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
    y[:, 0] = np.sum(x[:, :(n_features // 2)], axis=1)
    y[:, 1] = np.sum(x[:, (n_features // 2):], axis=1)
    y[:, 2] = 0.5 * np.sum(x, axis=1)
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(x, y, w)

    # Create an OntologyModel.  Two leaf nodes contain half the features.

    feature_ids = [str(i) for i in range(n_features)]
    node1 = dc.models.OntologyNode('leaf1', 5, feature_ids[:(n_features // 2)])
    node2 = dc.models.OntologyNode('leaf2', 5, feature_ids[(n_features // 2):])
    node3 = dc.models.OntologyNode('root', 5, children=[node1, node2])
    model = dc.models.OntologyModel(
        n_tasks, feature_ids, node3, learning_rate=0.02)

    # Train the model on the datase.

    model.fit(dataset, nb_epoch=1000)

    # It should have learned to predict all of the tasks accurately.

    pred_error = np.mean((model.predict_on_batch(x) - y)**2, axis=0)
    assert np.all(pred_error < 0.01)

    # In addition, it should be able to predict the first task based only on the
    # first leaf node, and the second task based only on the second leaf node.

    leaf1_pred = model.predict_on_batch(
        x, outputs=model.prediction_for_node['leaf1'])
    leaf2_pred = model.predict_on_batch(
        x, outputs=model.prediction_for_node['leaf2'])
    leaf1_error = np.mean((leaf1_pred - y)**2, axis=0)
    leaf2_error = np.mean((leaf2_pred - y)**2, axis=0)
    assert leaf1_error[0] < 0.01
    assert leaf2_error[1] < 0.01
    assert np.mean(
        [leaf1_error[1], leaf1_error[2], leaf2_error[0], leaf2_error[2]]) > 0.01

  def test_ontology_classifier(self):
    """Test training an OntologyModel for classification."""

    # Create a dataset with three tasks.  The first two tasks each depend only
    # on half the features.  The third task depends on all of them.

    n_features = 8
    n_tasks = 3
    n_classes = 2
    n_samples = 50
    x = np.random.random((n_samples, n_features))
    y = np.zeros((n_samples, n_tasks))
    y[:, 0] = np.sum(x[:, :(n_features // 2)], axis=1) < n_features / 4
    y[:, 1] = np.sum(x[:, (n_features // 2):], axis=1) < n_features / 4
    y[:, 2] = np.sum(x, axis=1) < n_features / 2
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(x, y, w)

    # Create an OntologyModel.  Two leaf nodes contain half the features.

    feature_ids = [str(i) for i in range(n_features)]
    node1 = dc.models.OntologyNode('leaf1', 5, feature_ids[:(n_features // 2)])
    node2 = dc.models.OntologyNode('leaf2', 5, feature_ids[(n_features // 2):])
    node3 = dc.models.OntologyNode('root', 5, children=[node1, node2])
    model = dc.models.OntologyModel(
        n_tasks, feature_ids, node3, mode='classification', learning_rate=0.02)

    # Train the model on the datase.

    model.fit(dataset, nb_epoch=1000)

    # It should have learned to predict all of the tasks accurately.

    pred = np.argmax(model.predict_on_batch(x), axis=2)
    pred_error = np.mean(np.abs(pred - y), axis=0)
    assert np.all(pred_error == 0)

    # In addition, it should be able to predict the first task based only on the
    # first leaf node, and the second task based only on the second leaf node.

    leaf1_pred = np.argmax(
        model.predict_on_batch(x, outputs=model.prediction_for_node['leaf1']),
        axis=2)
    leaf2_pred = np.argmax(
        model.predict_on_batch(x, outputs=model.prediction_for_node['leaf2']),
        axis=2)
    leaf1_error = np.mean(np.abs(leaf1_pred - y), axis=0)
    leaf2_error = np.mean(np.abs(leaf2_pred - y), axis=0)
    assert leaf1_error[0] == 0
    assert leaf2_error[1] == 0
    assert np.mean(
        [leaf1_error[1], leaf1_error[2], leaf2_error[0], leaf2_error[2]]) > 0.01

  def test_create_gene_ontology(self):
    """Test creating OntologyNodes for the Gene Ontology classification."""

    # Here are mappings for just a few yeast genes.

    mapping = {}
    mapping['STE7'] = ['GO:0000187']
    mapping['PBS2'] = ['GO:0000187']
    mapping['NOP8'] = [
        'GO:0003676', 'GO:0003723', 'GO:0042254', 'GO:0005634', 'GO:0005730'
    ]

    # Build the ontology, then see if it looks correct.

    root = dc.models.tensorgraph.models.ontology.create_gene_ontology(
        mapping, min_node_features=1)
    assert len(root.feature_ids) == 0

    def find_features(node, features):
      features.update(node.feature_ids)
      for child in node.children:
        find_features(child, features)

    all_features = set()
    find_features(root, all_features)
    assert len(all_features) == 3
    for key in mapping:
      assert key in all_features

"""Tests for the NequIP DeepChem integration scaffold."""

import numpy as np
import pytest
import torch

from deepchem.data import DiskDataset
from deepchem.feat.graph_data import BatchGraphData, GraphData
from deepchem.models.torch_models.nequip import NequIPModel

pytestmark = pytest.mark.torch


class _DummyModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.zeros(1))

    def forward(self, graph):
        batch_size = int(graph.graph_index.max().item()) + 1
        return self.value.expand(batch_size, 1)


def _dummy_loss(outputs, labels, weights):
    difference = outputs[0] - labels[0]
    return torch.mean(weights[0] * difference**2)


def _model(tasks, **kwargs):
    return NequIPModel(tasks=tasks,
                       model=_DummyModule(),
                       loss=_dummy_loss,
                       **kwargs)


def _graphs():
    graph_a = GraphData(node_features=np.array([[1], [8]], dtype=np.float32),
                        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
                        edge_features=np.array([[1, 0, 0], [-1, 0, 0]],
                                               dtype=np.float32),
                        node_pos_features=np.array([[0, 0, 0], [1, 0, 0]],
                                                   dtype=np.float32),
                        edge_distances=np.ones((2, 1), dtype=np.float32))
    graph_b = GraphData(node_features=np.array([[6], [1], [1]],
                                               dtype=np.float32),
                        edge_index=np.array([[0, 1, 2], [1, 2, 0]],
                                            dtype=np.int64),
                        edge_features=np.ones((3, 3), dtype=np.float32),
                        node_pos_features=np.array(
                            [[0, 0, 0], [0.8, 0, 0], [-0.8, 0, 0]],
                            dtype=np.float32),
                        edge_distances=np.ones((3, 1), dtype=np.float32))
    return [graph_a, graph_b]


def _forces():
    return [
        np.array([[0.1, 0, 0], [-0.1, 0, 0]], dtype=np.float32),
        np.array([[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]], dtype=np.float32)
    ]


def _labels(tasks):
    values = {"energy": [1.5, -2.0], "forces": _forces()}
    labels = np.empty((2, len(tasks)), dtype=object)
    for task_index, task in enumerate(tasks):
        labels[:, task_index] = values[task]
    return labels


def _prepare(tasks, weights):
    model = _model(tasks)
    return model._prepare_batch(([_graphs()], [_labels(tasks)], [weights]))


def _energy_dataset():
    graphs = np.asarray(_graphs(), dtype=object)
    return DiskDataset.create_dataset(iter([
        (graphs, _labels(["energy"]), np.ones(
            (2, 1), dtype=np.float32), np.array(["a", "b"]))
    ]),
                                      tasks=["energy"])


@pytest.mark.parametrize(
    "tasks",
    [["energy"], ["forces"], ["energy", "forces"], ["forces", "energy"]])
def test_task_order_is_supported_and_preserved(tasks):
    model = _model(tasks)
    assert model.tasks == tasks
    assert model.task_to_index == {
        task: index for index, task in enumerate(tasks)
    }


@pytest.mark.parametrize("tasks", [["energy", "energy"], ["dipole"]])
def test_invalid_tasks_are_rejected(tasks):
    with pytest.raises(ValueError):
        _model(tasks)


@pytest.mark.parametrize("model, loss", [(object(), _dummy_loss),
                                         (_DummyModule(), None)])
def test_injected_model_and_loss_are_validated(model, loss):
    with pytest.raises(TypeError):
        NequIPModel(tasks=["energy"], model=model, loss=loss)


def test_differently_sized_graphs_are_batched():
    graph, _, _ = _prepare(["energy"], np.ones((2, 1), dtype=np.float32))
    assert isinstance(graph, BatchGraphData)
    assert graph.node_features.shape == (5, 1)
    assert graph.node_pos_features.shape == (5, 3)
    assert graph.edge_index.shape == (2, 5)
    assert graph.edge_features.shape == (5, 3)
    assert graph.edge_distances.shape == (5, 1)
    assert graph.graph_index.tolist() == [0, 0, 1, 1, 1]
    assert graph.node_features.dtype == torch.float32
    assert graph.graph_index.dtype == torch.long


def test_energy_only_preparation():
    _, labels, weights = _prepare(["energy"],
                                  np.array([[0.5], [0.25]], dtype=np.float32))
    torch.testing.assert_close(labels[0], torch.tensor([[1.5], [-2.0]]))
    torch.testing.assert_close(weights[0], torch.tensor([[0.5], [0.25]]))
    assert labels[0].shape == (2, 1)
    assert weights[0].shape == (2, 1)
    assert labels[0].dtype == torch.float32
    assert weights[0].dtype == torch.float32


def test_force_only_preparation_and_weight_expansion():
    _, labels, weights = _prepare(["forces"],
                                  np.array([[0.3], [0.4]], dtype=np.float32))
    expected_forces = torch.tensor([[0.1, 0, 0], [-0.1, 0, 0], [0.2, 0, 0],
                                    [0, 0.2, 0], [0, 0, 0.2]])
    torch.testing.assert_close(labels[0], expected_forces)
    torch.testing.assert_close(
        weights[0], torch.tensor([[0.3], [0.3], [0.4], [0.4], [0.4]]))
    assert labels[0].shape == (5, 3)
    assert weights[0].shape == (5, 1)
    assert labels[0].dtype == torch.float32
    assert weights[0].dtype == torch.float32


def test_energy_and_forces_are_prepared_together():
    _, labels, weights = _prepare(["energy", "forces"],
                                  np.array([[0.5, 0.3], [0.25, 0.4]],
                                           dtype=np.float32))
    torch.testing.assert_close(labels[0], torch.tensor([[1.5], [-2.0]]))
    torch.testing.assert_close(labels[1][:2],
                               torch.tensor([[0.1, 0, 0], [-0.1, 0, 0]]))
    assert labels[0].shape == (2, 1)
    assert labels[1].shape == (5, 3)
    assert weights[0].shape == (2, 1)
    assert weights[1].shape == (5, 1)


def test_reversed_task_order_uses_correct_columns():
    _, labels, weights = _prepare(["forces", "energy"],
                                  np.array([[0.3, 0.5], [0.4, 0.25]],
                                           dtype=np.float32))
    torch.testing.assert_close(labels[0][:2],
                               torch.tensor([[0.1, 0, 0], [-0.1, 0, 0]]))
    torch.testing.assert_close(labels[1], torch.tensor([[1.5], [-2.0]]))
    torch.testing.assert_close(
        weights[0], torch.tensor([[0.3], [0.3], [0.4], [0.4], [0.4]]))
    torch.testing.assert_close(weights[1], torch.tensor([[0.5], [0.25]]))


def test_invalid_force_shape_identifies_sample_and_expected_shape():
    labels = _labels(["forces"])
    labels[1, 0] = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError,
                       match=r"sample 1.*shape \(2, 3\).*Expected \(3, 3\)"):
        _model(["forces"])._prepare_batch(
            ([_graphs()], [labels], [np.ones((2, 1), dtype=np.float32)]))


def test_non_scalar_energy_is_rejected():
    labels = _labels(["energy"])
    labels[1, 0] = np.array([-2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="Energy label for sample 1.*scalar"):
        _model(["energy"])._prepare_batch(
            ([_graphs()], [labels], [np.ones((2, 1), dtype=np.float32)]))


def test_default_generator_never_pads_final_graph_batch():
    dataset = _energy_dataset()
    inputs, labels, weights = next(
        _model(["energy"], batch_size=4).default_generator(dataset,
                                                           pad_batches=True))
    assert len(inputs[0]) == 2
    assert labels[0].shape == (2, 1)
    assert weights[0].shape == (2, 1)


def test_unlabeled_diskdataset_batch_is_prepared_without_properties():
    graphs = np.asarray(_graphs(), dtype=object)
    dataset = DiskDataset.create_dataset(iter([(graphs, None, None,
                                                np.array(["a", "b"]))]),
                                         tasks=[])
    generated_batch = next(_model([], batch_size=4).default_generator(dataset))
    graph, labels, weights = _model([])._prepare_batch(generated_batch)
    assert graph.graph_index.tolist() == [0, 0, 1, 1, 1]
    assert generated_batch[1] == [None]
    assert generated_batch[2] == [None]
    assert labels == []
    assert weights == []


def test_prepared_values_use_model_device():
    model = _model(["energy"], device=torch.device("cpu"))
    graph, labels, weights = model._prepare_batch(
        ([_graphs()], [_labels(["energy"])],
         [np.ones((2, 1), dtype=np.float32)]))
    assert graph.node_features.device.type == "cpu"
    assert graph.node_pos_features.device.type == "cpu"
    assert labels[0].device.type == "cpu"
    assert weights[0].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_batch_preparation():
    model = _model(["forces"], device=torch.device("cuda"))
    graph, labels, weights = model._prepare_batch(
        ([_graphs()], [_labels(["forces"])],
         [np.ones((2, 1), dtype=np.float32)]))
    assert graph.node_features.device.type == "cuda"
    assert labels[0].device.type == "cuda"
    assert weights[0].device.type == "cuda"


def test_energy_fit_runs_through_torchmodel():
    model = _model(["energy"], batch_size=4, learning_rate=0.1)
    initial_value = model.model.value.detach().clone()
    loss = model.fit(_energy_dataset(),
                     nb_epoch=1,
                     checkpoint_interval=0,
                     deterministic=True)
    assert np.isfinite(loss)
    assert not torch.equal(model.model.value.detach(), initial_value)


def test_public_export():
    from deepchem.models.torch_models import NequIPModel as ExportedNequIPModel

    assert ExportedNequIPModel is NequIPModel

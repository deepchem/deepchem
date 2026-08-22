"""Tests for the native NequIP backbone."""

import copy
import math

import numpy as np
import pytest
import torch

from deepchem.feat.graph_data import BatchGraphData, GraphData
from deepchem.models.torch_models.nequip import (NequIP,
                                                 _TensorProductConvolution)
from deepchem.utils.equivariance_utils import (get_spherical_from_cartesian,
                                               precompute_sh)


def _graph(positions, atomic_numbers):
    positions = np.asarray(positions, dtype=np.float32)
    num_atoms = len(positions)
    center = []
    neighbor = []
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                center.append(i)
                neighbor.append(j)
    edge_index = np.asarray([center, neighbor], dtype=np.int64)
    edge_vectors = positions[edge_index[1]] - positions[edge_index[0]]
    edge_distances = np.linalg.norm(edge_vectors, axis=1, keepdims=True)
    return GraphData(node_features=np.asarray(atomic_numbers,
                                              dtype=np.int64).reshape(-1, 1),
                     edge_index=edge_index,
                     edge_features=edge_vectors.copy(),
                     node_pos_features=positions,
                     edge_distances=edge_distances)


def _graphs():
    first = _graph([[0.1, 0.2, -0.1], [1.0, -0.3, 0.4]], [1, 8])
    second = _graph([[0.2, -0.4, 0.3], [1.1, 0.1, -0.2], [-0.5, 0.8, 0.7]],
                    [6, 1, 8])
    return first, second


def _batch(*graphs):
    return BatchGraphData(list(graphs)).numpy_to_torch()


@pytest.fixture
def model():
    torch.manual_seed(13)
    return NequIP(cutoff=4.0,
                  num_species=10,
                  hidden_channels=3,
                  num_interaction_blocks=3,
                  l_max=1,
                  num_bessel=4,
                  radial_hidden_width=8,
                  radial_hidden_depth=1,
                  avg_num_neighbors=2.0,
                  readout_hidden_channels=3)


def _transformed_batch(graphs, matrix=None, translation=None):
    transformed = []
    for graph in graphs:
        graph = copy.deepcopy(graph)
        positions = graph.node_pos_features
        if matrix is not None:
            positions = positions @ matrix.T
        if translation is not None:
            positions = positions + translation
        graph.node_pos_features = positions
        transformed.append(graph)
    return _batch(*transformed)


def test_tensor_product_paths_and_multiplicities():
    e, o = 1, -1
    expected = {
        ((0, e), 0): {(0, e)},
        ((0, e), 1): {(1, o)},
        ((0, e), 2): {(2, e)},
        ((0, o), 0): {(0, o)},
        ((0, o), 1): {(1, e)},
        ((0, o), 2): {(2, o)},
        ((1, e), 0): {(1, e)},
        ((1, e), 1): {(0, o), (1, o), (2, o)},
        ((1, e), 2): {(1, e), (2, e)},
        ((1, o), 0): {(1, o)},
        ((1, o), 1): {(0, e), (1, e), (2, e)},
        ((1, o), 2): {(1, o), (2, o)},
        ((2, e), 0): {(2, e)},
        ((2, e), 1): {(1, o), (2, o)},
        ((2, e), 2): {(0, e), (1, e), (2, e)},
        ((2, o), 0): {(2, o)},
        ((2, o), 1): {(1, e), (2, e)},
        ((2, o), 2): {(0, o), (1, o), (2, o)},
    }
    input_irreps = {
        (0, e): 2,
        (0, o): 3,
        (1, e): 4,
        (1, o): 5,
        (2, e): 6,
        (2, o): 7,
    }
    convolution = _TensorProductConvolution(
        input_irreps=input_irreps,
        output_irreps={irrep: 2 for irrep in input_irreps},
        l_max=2,
        num_bessel=3,
        radial_hidden_width=4,
        radial_hidden_depth=1,
        avg_num_neighbors=1.0)
    actual = {}
    expected_mid = {irrep: 0 for irrep in input_irreps}
    for path in convolution.paths:
        actual.setdefault((path.input_irrep, path.filter_degree),
                          set()).add(path.output_irrep)
        path_channels = path.weight_stop - path.weight_start
        assert path_channels == input_irreps[path.input_irrep]
        expected_mid[path.output_irrep] += path_channels
    assert actual == expected
    assert convolution.post_self_interaction.input_irreps == expected_mid
    expected_weights = sum(input_irreps[input_irrep] * len(outputs)
                           for (input_irrep, _), outputs in expected.items())
    assert convolution.paths[-1].weight_stop == expected_weights
    assert convolution.radial_mlp.layers[-1].out_features == expected_weights


def test_tensor_product_component_normalization():
    direction = torch.tensor([[0.3, -0.4, 0.5]])
    direction = direction / torch.linalg.vector_norm(direction, dim=-1)
    spherical_harmonics = precompute_sh(get_spherical_from_cartesian(direction),
                                        2)
    radial_basis = torch.ones(1, 3)
    features = {(0, 1): torch.ones(2, 1, 1)}

    for output_degree in range(3):
        output_irrep = (output_degree, 1 if output_degree % 2 == 0 else -1)
        convolution = _TensorProductConvolution(input_irreps={(0, 1): 1},
                                                output_irreps={output_irrep: 1},
                                                l_max=2,
                                                num_bessel=3,
                                                radial_hidden_width=4,
                                                radial_hidden_depth=1,
                                                avg_num_neighbors=1.0)
        interaction = convolution.post_self_interaction.interactions[str(
            output_irrep[1])]
        with torch.no_grad():
            interaction.transform[str(output_degree)].fill_(1.0)
        radial_weight = convolution.radial_mlp(radial_basis)[0, 0]
        output = convolution(features,
                             radial_basis,
                             spherical_harmonics,
                             center=torch.tensor([0]),
                             neighbor=torch.tensor([1]),
                             num_nodes=2)[output_irrep][0, 0]
        expected_norm = radial_weight.abs() * math.sqrt(2 * output_degree + 1)
        torch.testing.assert_close(torch.linalg.vector_norm(output),
                                   expected_norm)


def test_unequal_batch_matches_individual_graphs(model):
    first, second = _graphs()
    with torch.no_grad():
        batched_energy = model(_batch(first, second))
        first_energy = model(_batch(first))
        second_energy = model(_batch(second))
    assert batched_energy.shape == (2, 1)
    torch.testing.assert_close(batched_energy,
                               torch.cat([first_energy, second_energy]))


def test_translation_invariance_and_force_invariance(model):
    graphs = _graphs()
    original_energy, original_forces = model(_batch(*graphs),
                                             compute_forces=True)
    translated = _transformed_batch(graphs,
                                    translation=np.asarray([2.1, -0.7, 1.3],
                                                           dtype=np.float32))
    translated_energy, translated_forces = model(translated,
                                                 compute_forces=True)
    torch.testing.assert_close(translated_energy,
                               original_energy,
                               atol=2e-5,
                               rtol=2e-5)
    torch.testing.assert_close(translated_forces,
                               original_forces,
                               atol=3e-5,
                               rtol=3e-5)


def test_rotation_invariance_and_force_equivariance(model):
    graphs = _graphs()
    angle = 0.73
    axis = torch.tensor([0.3, -0.5, 0.8])
    axis = axis / torch.linalg.vector_norm(axis)
    cross = torch.tensor([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]],
                          [-axis[1], axis[0], 0.0]])
    rotation = (
        torch.eye(3) * torch.cos(torch.tensor(angle)) +
        (1.0 - torch.cos(torch.tensor(angle))) * torch.outer(axis, axis) +
        torch.sin(torch.tensor(angle)) * cross)

    original_energy, original_forces = model(_batch(*graphs),
                                             compute_forces=True)
    rotated_energy, rotated_forces = model(_transformed_batch(
        graphs, matrix=rotation.numpy()),
                                           compute_forces=True)
    torch.testing.assert_close(rotated_energy,
                               original_energy,
                               atol=3e-4,
                               rtol=3e-4)
    torch.testing.assert_close(rotated_forces,
                               original_forces @ rotation.T,
                               atol=5e-4,
                               rtol=5e-4)


def test_inversion_invariance_and_force_equivariance(model):
    graphs = _graphs()
    inversion = -np.eye(3, dtype=np.float32)
    original_energy, original_forces = model(_batch(*graphs),
                                             compute_forces=True)
    inverted_energy, inverted_forces = model(_transformed_batch(
        graphs, matrix=inversion),
                                             compute_forces=True)
    torch.testing.assert_close(inverted_energy,
                               original_energy,
                               atol=3e-5,
                               rtol=3e-5)
    torch.testing.assert_close(inverted_forces,
                               -original_forces,
                               atol=5e-5,
                               rtol=5e-5)


def test_force_is_negative_energy_gradient(model):
    graphs = _graphs()
    returned_energy, returned_force = model(_batch(*graphs),
                                            compute_forces=True)
    independent_batch = _batch(*graphs)
    independent_batch.node_pos_features.requires_grad_(True)
    independent_energy = model(independent_batch)
    independent_force = -torch.autograd.grad(
        independent_energy.sum(), independent_batch.node_pos_features)[0]
    torch.testing.assert_close(returned_energy, independent_energy)
    torch.testing.assert_close(returned_force, independent_force)


def test_force_matches_finite_difference(model):
    first, _ = _graphs()
    model = copy.deepcopy(model).double()
    batch = _batch(first)
    batch.node_pos_features = batch.node_pos_features.double()
    _, forces = model(batch, compute_forces=True)

    epsilon = 1e-5
    displaced_energies = []
    for displacement in (epsilon, -epsilon):
        displaced = _batch(first)
        displaced.node_pos_features = displaced.node_pos_features.double()
        displaced.node_pos_features[0, 1] += displacement
        displaced_energies.append(model(displaced))
    finite_difference = -(displaced_energies[0] - displaced_energies[1]) / (
        2 * epsilon)
    torch.testing.assert_close(forces[0, 1],
                               finite_difference[0, 0],
                               atol=1e-6,
                               rtol=1e-4)


def test_isolated_atom_has_finite_energy_and_zero_force(model):
    isolated = GraphData(node_features=np.asarray([[1]], dtype=np.int64),
                         edge_index=np.empty((2, 0), dtype=np.int64),
                         edge_features=np.empty((0, 3), dtype=np.float32),
                         node_pos_features=np.asarray([[0.2, -0.1, 0.7]],
                                                      dtype=np.float32),
                         edge_distances=np.empty((0, 1), dtype=np.float32))
    batch = _batch(isolated)
    energy, force = model(batch, compute_forces=True)
    assert energy.shape == (1, 1)
    assert force.shape == (1, 3)
    assert torch.isfinite(energy).all()
    assert torch.isfinite(force).all()
    torch.testing.assert_close(force, torch.zeros_like(force))
    assert not batch.node_pos_features.requires_grad


def test_force_loss_is_differentiable(model):
    _, forces = model(_batch(*_graphs()),
                      compute_forces=True,
                      create_graph=True)
    forces.square().mean().backward()
    gradients = [
        model.embedding.weight.grad, model.bessel_basis.frequencies.grad,
        model.interaction_blocks[0].convolution.radial_mlp.layers[0].weight.grad
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_static_geometry_is_ignored(model):
    batch = _batch(*_graphs())
    changed = copy.deepcopy(batch)
    changed.edge_features = torch.randn_like(changed.edge_features) * 1000
    changed.edge_distances = torch.randn_like(changed.edge_distances) * 1000
    with torch.no_grad():
        expected = model(batch)
        actual = model(changed)
    torch.testing.assert_close(actual, expected)

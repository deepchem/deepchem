from deepchem.feat import Featurizer
from typing import Any, List
import numpy as np

class DTNNFeaturizer(Featurizer):
    def __init__(self, steps, step_size):
        self.steps = steps
        self.step_size = step_size

    def _featurize(self, datapoint: np.array, **kwargs):
        distance = []
        atom_membership = []
        distance_membership_i = []
        distance_membership_j = []
        num_atoms = list(map(sum, datapoint.astype(bool)[:, :, 0]))
        atom_number = [
            np.round(
                np.power(2 * np.diag(datapoint[i, :num_atoms[i], :num_atoms[i]]),
                         1 / 2.4)).astype(int) for i in range(len(num_atoms))
        ]
        start = 0
        for im, molecule in enumerate(atom_number):
            distance_matrix = np.outer(
                molecule, molecule) / datapoint[im, :num_atoms[im], :num_atoms[im]]
            np.fill_diagonal(distance_matrix, -100)
            distance.append(np.expand_dims(distance_matrix.flatten(), 1))
            atom_membership.append([im] * num_atoms[im])
            membership = np.array([np.arange(num_atoms[im])] * num_atoms[im])
            membership_i = membership.flatten(order='F')
            membership_j = membership.flatten()
            distance_membership_i.append(membership_i + start)
            distance_membership_j.append(membership_j + start)
            start = start + num_atoms[im]

        atom_number = np.concatenate(atom_number).astype(np.int32)
        distance = np.concatenate(distance, axis=0)
        gaussian_dist = np.exp(-np.square(distance - self.steps) /
                               (2 * self.step_size**2))
        gaussian_dist = gaussian_dist.astype(np.float64)
        atom_mem = np.concatenate(atom_membership).astype(np.int64)
        dist_mem_i = np.concatenate(distance_membership_i).astype(np.int64)
        dist_mem_j = np.concatenate(distance_membership_j).astype(np.int64)

        features = [
            atom_number, gaussian_dist, atom_mem, dist_mem_i, dist_mem_j
        ]
        return features
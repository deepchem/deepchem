"""
Utility Functions for computing features on batch.
"""
import numpy as np


def coulomb_matrix_features(X_b: np.ndarray,
                            distance_max: float = -1,
                            distance_min: float = 18,
                            n_distance: int = 100):
    """Computes the values for different Feature Layers on given batch.
    It works as a helper function to coulomb matrix.

    Parameters
    ----------
    X_b: np.ndarray
        It is a 3d Matrix containing information of each the atom's ionic interaction with other atoms in the molecule.
    distance_min: float (default -1)
        minimum distance of atom pairs (in Angstrom)
    distance_max: float (default = 18)
        maximum distance of atom pairs (in Angstrom)
    n_distance: int (default 100)
        granularity of distance matrix
        step size will be (distance_max-distance_min)/n_distance

    Returns
    -------
    atom_number: np.ndarray
        Atom numbers are assigned to each atom based on their atomic properties.
        The atomic numbers are derived from the periodic table of elements.
        For example, hydrogen -> 1, carbon -> 6, and oxygen -> 8.
    gaussian_dist: np.ndarray
        Gaussian distance refers to the method of representing the pairwise distances between atoms in a molecule using Gaussian functions.
        The Gaussian distance is calculated using the Euclidean distance between the Cartesian coordinates of two atoms.
        The distance value is then passed through a Gaussian function, which transforms it into a continuous value.
    atom_mem: np.ndarray
        Atom membership refers to the binary representation of whether an atom belongs to a specific group or property within a molecule.
        It allows the model to incorporate domain-specific information and enhance its understanding of the molecule's properties and interactions.
    dist_mem_i: np.ndarray
        Distance membership i are utilized to encode spatial information and capture the influence of atom distances on the properties and interactions within a molecule.
        The inner membership function assigns higher values to atoms that are closer to the atoms' interaction region, thereby emphasizing the impact of nearby atoms.
    dist_mem_j: np.ndarray
        It captures the long-range effects and influences between atoms that are not in direct proximity but still contribute to the overall molecular properties.
        Distance membership j are utilized to encode spatial information and capture the influence of atom distances on the properties and interactions outside a molecule.
        The outer membership function assigns higher values to atoms that are farther to the atoms' interaction region, thereby emphasizing the impact of farther atoms.

    """
    distance = []
    atom_membership = []
    distance_membership_i = []
    distance_membership_j = []

    # Calculation of Step Size and steps
    step_size = (distance_max - distance_min) / n_distance
    steps = np.array([distance_min + i * step_size for i in range(n_distance)])
    steps = np.expand_dims(steps, 0)

    # Number of atoms per molecule is calculated by counting all the non zero elements(numbers) of every molecule.
    num_atoms = list(map(sum, X_b.astype(bool)[:, :, 0]))

    # It loops over the molecules in the Coulomb matrix and rounds the square root of the diagonal of each molecule to the nearest integer.
    atom_number = [
        np.round(
            np.power(2 * np.diag(X_b[i, :num_atoms[i], :num_atoms[i]]),
                     1 / 2.4)).astype(int) for i in range(len(num_atoms))
    ]
    start = 0
    for im, molecule in enumerate(atom_number):
        distance_matrix = np.outer(
            molecule, molecule) / X_b[im, :num_atoms[im], :num_atoms[im]]
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

    # Calculates the Gaussian Distance by passing distance by a gaussian function.
    gaussian_dist = np.exp(-np.square(distance - steps) / (2 * step_size**2))
    gaussian_dist = gaussian_dist.astype(np.float64)
    atom_mem = np.concatenate(atom_membership).astype(np.int64)
    dist_mem_i = np.concatenate(distance_membership_i).astype(np.int64)
    dist_mem_j = np.concatenate(distance_membership_j).astype(np.int64)
    features = [atom_number, gaussian_dist, atom_mem, dist_mem_i, dist_mem_j]
    return features

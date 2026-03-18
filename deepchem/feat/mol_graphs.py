"""
Data Structures used to represented molecules for convolutions.
"""
# flake8: noqa

import csv
import random
import numpy as np


def cumulative_sum_minus_last(l, offset=0):
    """Returns cumulative sums for set of counts, removing last entry.

    Returns the cumulative sums for a set of counts with the first returned value
    starting at 0. I.e [3,2,4] -> [0, 3, 5]. Note last sum element 9 is missing.
    Useful for reindexing

    Parameters
    ----------
    l: list
        List of integers. Typically small counts.
    """
    return np.delete(np.insert(np.cumsum(l, dtype=np.int32), 0, 0), -1) + offset


def cumulative_sum(l, offset=0):
    """Returns cumulative sums for set of counts.

    Returns the cumulative sums for a set of counts with the first returned value
    starting at 0. I.e [3,2,4] -> [0, 3, 5, 9]. Keeps final sum for searching.
    Useful for reindexing.

    Parameters
    ----------
    l: list
        List of integers. Typically small counts.
    """
    return np.insert(np.cumsum(l), 0, 0) + offset


class ConvMol(object):
    """Holds information about a molecules.

    Resorts order of atoms internally to be in order of increasing degree. Note
    that only heavy atoms (hydrogens excluded) are considered here.
    """

    def __init__(self, atom_features, adj_list, max_deg=10, min_deg=0):
        """
        Parameters
        ----------
        atom_features: np.ndarray
            Has shape (n_atoms, n_feat)
        adj_list: list
            List of length n_atoms, with neighor indices of each atom.
        max_deg: int, optional
            Maximum degree of any atom.
        min_deg: int, optional
            Minimum degree of any atom.
        """

        self.atom_features = atom_features
        self.n_atoms, self.n_feat = atom_features.shape
        self.deg_list = np.array([len(nbrs) for nbrs in adj_list],
                                 dtype=np.int32)
        self.canon_adj_list = adj_list
        self.deg_adj_lists = []
        self.deg_slice = []
        self.max_deg = max_deg
        self.min_deg = min_deg

        self.membership = self.get_num_atoms() * [0]

        self._deg_sort()

        # Get the degree id list (which corrects for min_deg)
        self.deg_id_list = np.array(self.deg_list) - min_deg

        # Get the size of each degree block
        deg_size = [
            self.get_num_atoms_with_deg(deg)
            for deg in range(self.min_deg, self.max_deg + 1)
        ]

        self.degree_list = []
        for i, deg in enumerate(range(self.min_deg, self.max_deg + 1)):
            self.degree_list.extend([deg] * deg_size[i])

        # Get the the start indices for items in each block
        self.deg_start = cumulative_sum(deg_size)

        # Get the node indices when they are reset when the degree changes
        deg_block_indices = [
            i - self.deg_start[self.deg_list[i]] for i in range(self.n_atoms)
        ]

        # Convert to numpy array
        self.deg_block_indices = np.array(deg_block_indices, dtype=np.int32)

    def get_atoms_with_deg(self, deg):
        """Retrieves atom_features with the specific degree"""
        start_ind = self.deg_slice[deg - self.min_deg, 0]
        size = self.deg_slice[deg - self.min_deg, 1]
        return self.atom_features[start_ind:(start_ind + size), :]

    def get_num_atoms_with_deg(self, deg):
        """Returns the number of atoms with the given degree"""
        return self.deg_slice[deg - self.min_deg, 1]

    def get_num_atoms(self):
        return self.n_atoms

    def _deg_sort(self):
        """Sorts atoms by degree and reorders internal data structures.
    
        Sort the order of the atom_features by degree, maintaining original order
        whenever two atom_features have the same degree.
        """
        old_ind = range(self.get_num_atoms())
        deg_list = self.deg_list
        new_ind = list(np.lexsort((old_ind, deg_list)))

        num_atoms = self.get_num_atoms()

        # Reorder old atom_features
        self.atom_features = self.atom_features[new_ind, :]

        # Reorder old deg lists
        self.deg_list = [self.deg_list[i] for i in new_ind]

        # Sort membership
        self.membership = [self.membership[i] for i in new_ind]

        # Create old to new dictionary. not exactly intuitive
        old_to_new = dict(zip(new_ind, old_ind))

        # Reorder adjacency lists
        self.canon_adj_list = [self.canon_adj_list[i] for i in new_ind]
        self.canon_adj_list = [[old_to_new[k]
                                for k in self.canon_adj_list[i]]
                               for i in range(len(new_ind))]

        # Get numpy version of degree list for indexing
        deg_array = np.array(self.deg_list)

        # Initialize adj_lists, which supports min_deg = 1 only
        self.deg_adj_lists = (self.max_deg + 1 - self.min_deg) * [0]

        # Parse as deg separated
        for deg in range(self.min_deg, self.max_deg + 1):
            # Get indices corresponding to the current degree
            rng = np.array(range(num_atoms))
            indices = rng[deg_array == deg]

            # Extract and save adjacency list for the current degree
            to_cat = [self.canon_adj_list[i] for i in indices]
            if len(to_cat) > 0:
                adj_list = np.vstack([self.canon_adj_list[i] for i in indices])
                self.deg_adj_lists[deg - self.min_deg] = adj_list.astype(
                    np.int32)

            else:
                self.deg_adj_lists[deg - self.min_deg] = np.zeros(
                    [0, deg], dtype=np.int32)

        # Construct the slice information
        deg_slice = np.zeros([self.max_deg + 1 - self.min_deg, 2],
                             dtype=np.int32)

        for deg in range(self.min_deg, self.max_deg + 1):
            if deg == 0:
                deg_size = np.sum(deg_array == deg)
            else:
                deg_size = self.deg_adj_lists[deg - self.min_deg].shape[0]

            deg_slice[deg - self.min_deg, 1] = deg_size
            # Get the cumulative indices after the first index
            if deg > self.min_deg:
                deg_slice[deg - self.min_deg,
                          0] = (deg_slice[deg - self.min_deg - 1, 0] +
                                deg_slice[deg - self.min_deg - 1, 1])

        # Set indices with zero sized slices to zero to avoid indexing errors
        deg_slice[:, 0] *= (deg_slice[:, 1] != 0)
        self.deg_slice = deg_slice

    def get_atom_features(self):
        """Returns canonicalized version of atom features.
    
        Features are sorted by atom degree, with original order maintained when
        degrees are same.
        """
        return self.atom_features

    def get_adjacency_list(self):
        """Returns a canonicalized adjacency list.
    
        Canonicalized means that the atoms are re-ordered by degree.
    
        Returns
        -------
        list
            Canonicalized form of adjacency list.
        """
        return self.canon_adj_list

    def get_deg_adjacency_lists(self):
        """Returns adjacency lists grouped by atom degree.
    
        Returns
        -------
        list
            Has length (max_deg+1-min_deg). The element at position deg is
            itself a list of the neighbor-lists for atoms with degree deg.
        """
        return self.deg_adj_lists

    def get_deg_slice(self):
        """Returns degree-slice tensor.
    
        The deg_slice tensor allows indexing into a flattened version of the
        molecule's atoms. Assume atoms are sorted in order of degree. Then
        deg_slice[deg][0] is the starting position for atoms of degree deg in
        flattened list, and deg_slice[deg][1] is the number of atoms with degree deg.
    
        Note deg_slice has shape (max_deg+1-min_deg, 2).
    
        Returns
        -------
        deg_slice: np.ndarray
            Shape (max_deg+1-min_deg, 2)
        """
        return self.deg_slice

    # TODO(rbharath): Can this be removed?
    @staticmethod
    def get_null_mol(n_feat, max_deg=10, min_deg=0):
        """Constructs a null molecules
    
        Get one molecule with one atom of each degree, with all the atoms
        connected to themselves, and containing n_feat features.
    
        Parameters
        ----------
        n_feat : int
            number of features for the nodes in the null molecule
        """
        # Use random insted of zeros to prevent weird issues with summing to zero
        atom_features = np.random.uniform(0, 1, [max_deg + 1 - min_deg, n_feat])
        canon_adj_list = [
            deg * [deg - min_deg] for deg in range(min_deg, max_deg + 1)
        ]

        return ConvMol(atom_features, canon_adj_list)

    @staticmethod
    def agglomerate_mols(mols, max_deg=10, min_deg=0):
        """Concatenates list of ConvMol's into one mol object that can be used to feed
            into tensorflow placeholders. The indexing of the molecules are preseved during the
            combination, but the indexing of the atoms are greatly changed.
    
        Parameters
        ----------
        mols: list
            ConvMol objects to be combined into one molecule.
        """

        num_mols = len(mols)

        # Combine the features, then sort them by (atom_degree, mol_index)
        atoms_by_deg = np.concatenate([x.atom_features for x in mols])
        degree_vector = np.concatenate([x.degree_list for x in mols], axis=0)
        # Mergesort is a "stable" sort, so the array maintains it's secondary sort of mol_index
        order = degree_vector.argsort(kind='mergesort')
        ordered = np.empty(order.shape, np.int32)
        ordered[order] = np.arange(order.shape[0], dtype=np.int32)
        all_atoms = atoms_by_deg[order]

        # Create a map from the original atom indices within each molecule to the
        # indices in the combined object.
        mol_atom_map = []
        index_start = 0
        for mol in mols:
            mol_atom_map.append(ordered[index_start:index_start +
                                        mol.get_num_atoms()])
            index_start += mol.get_num_atoms()

        # Sort all atoms by degree.
        # Get the size of each atom list separated by molecule id, then by degree
        mol_deg_sz = np.zeros([max_deg - min_deg + 1, num_mols], dtype=np.int32)
        for i, mol in enumerate(mols):
            mol_deg_sz[:, i] += mol.deg_slice[:, 1]

        # Get the final size of each degree block
        deg_sizes = np.sum(mol_deg_sz, axis=1)

        # Get the index at which each degree starts, not resetting after each degree
        # And not stopping at any specific molecule

        deg_start = cumulative_sum_minus_last(deg_sizes)

        # Get the tensorflow object required for slicing (deg x 2) matrix, with the
        # first column telling the start indices of each degree block and the
        # second colum telling the size of each degree block
        deg_slice = np.array(list(zip(deg_start, deg_sizes)))

        # Determine the membership (atom i belongs to molecule membership[i])
        membership = np.empty(all_atoms.shape[0], np.int32)
        for i in range(num_mols):
            membership[mol_atom_map[i]] = i

        # Initialize the new degree separated adjacency lists
        deg_adj_lists = [
            np.empty([deg_sizes[deg], deg], dtype=np.int32)
            for deg in range(min_deg, max_deg + 1)
        ]

        # Update the old adjacency lists with the new atom indices and then combine
        # all together
        for deg in range(min_deg, max_deg + 1):
            row = 0  # Initialize counter
            deg_id = deg - min_deg  # Get corresponding degree id

            # Iterate through all the molecules
            for mol_id in range(num_mols):
                # Get the adjacency lists for this molecule and current degree id
                nbr_list = mols[mol_id].deg_adj_lists[deg_id]

                # Correct all atom indices to the final indices, and then save the
                # results into the new adjacency lists
                if nbr_list.shape[0] > 0:
                    if nbr_list.dtype == np.int32:
                        final_id = mol_atom_map[mol_id][nbr_list]
                        deg_adj_lists[deg_id][row:(
                            row + nbr_list.shape[0])] = final_id
                        row += nbr_list.shape[0]
                    else:
                        for i in range(nbr_list.shape[0]):
                            for j in range(nbr_list.shape[1]):
                                deg_adj_lists[deg_id][
                                    row, j] = mol_atom_map[mol_id][nbr_list[i,
                                                                            j]]
                            # Increment once row is done
                            row += 1

        # Get the final aggregated molecule
        concat_mol = MultiConvMol(all_atoms, deg_adj_lists, deg_slice,
                                  membership, num_mols)
        return concat_mol


class MultiConvMol(object):
    """Holds information about multiple molecules, for use in feeding information
        into tensorflow. Generated using the agglomerate_mols function
    """

    def __init__(self, nodes, deg_adj_lists, deg_slice, membership, num_mols):
        self.nodes = nodes
        self.deg_adj_lists = deg_adj_lists
        self.deg_slice = deg_slice
        self.membership = membership
        self.num_mols = num_mols
        self.num_atoms = nodes.shape[0]

    def get_deg_adjacency_lists(self):
        return self.deg_adj_lists

    def get_atom_features(self):
        return self.nodes

    def get_num_atoms(self):
        return self.num_atoms

    def get_num_molecules(self):
        return self.num_mols


class WeaveMol(object):
    """Molecular featurization object for weave convolutions.

    These objects are produced by WeaveFeaturizer, and feed into
    WeaveModel. The underlying implementation is inspired by [1]_.


    References
    ----------
    .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond fingerprints." Journal of computer-aided molecular design 30.8 (2016): 595-608.
    """

    def __init__(self, nodes, pairs, pair_edges):
        self.nodes = nodes
        self.pairs = pairs
        self.num_atoms = self.nodes.shape[0]
        self.n_features = self.nodes.shape[1]
        self.pair_edges = pair_edges

    def get_pair_edges(self):
        return self.pair_edges

    def get_pair_features(self):
        return self.pairs

    def get_atom_features(self):
        return self.nodes

    def get_num_atoms(self):
        return self.num_atoms

    def get_num_features(self):
        return self.n_features

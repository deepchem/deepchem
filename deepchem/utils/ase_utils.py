# -*- coding: utf-8 -*-
import numpy as np
import torch
from deepchem.feat import EquivariantGraphFeaturizer

try:
    from ase.calculators.calculator import Calculator, all_changes
    from ase import Atoms
except ImportError:
    pass

class DeepChemASECalculator(Calculator):
    """
    ASE Calculator that uses a DeepChem model for energy and forces.
    
    Parameters
    ----------
    model : deepchem.models.TorchModel
        The trained DeepChem model.
    featurizer : deepchem.feat.MolecularFeaturizer
        Featurizer to convert ASE Atoms to DeepChem GraphData.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, model, featurizer, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.model = model
        self.featurizer = featurizer

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """
        Calculate energy and forces.
        """
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # Convert ASE Atoms to GraphData
        z = atoms.get_atomic_numbers()
        pos = atoms.get_positions()
        graph_data = self._atoms_to_graph_data(z, pos)
        
        # Create dataset and run prediction
        import deepchem as dc
        dataset = dc.data.NumpyDataset(X=[graph_data])
        outputs = self.model.predict(dataset)
        
        if isinstance(outputs, list):
             energy = outputs[0]
             forces = outputs[1]
        else:
             energy = outputs
             forces = np.zeros_like(pos)
             
        self.results['energy'] = float(energy)
        self.results['forces'] = np.array(forces)

    def _atoms_to_graph_data(self, z, pos, cutoff=5.0):
        """
        Convert atomic numbers and positions to GraphData with a distance-based cutoff.
        """
        from deepchem.feat.graph_data import GraphData
        
        num_atoms = len(z)
        
        # Compute adjacency based on distance cutoff
        dist_matrix = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        adj = (dist_matrix < cutoff) & (dist_matrix > 0)
        src, dst = np.where(adj)
        
        # Encode features: Atomic Numbers in the last column
        node_feats = np.zeros((num_atoms, 2))
        node_feats[:, -1] = z 
        
        edge_index = np.array([src, dst])
        
        # Compute relative positions (edge features) for Equivariance
        # r_ij = r_dst - r_src
        # Shape: (num_edges, 3)
        edge_vecs = pos[dst] - pos[src]
        
        return GraphData(node_features=node_feats,
                         edge_index=edge_index,
                         edge_features=edge_vecs,
                         node_pos_features=pos)

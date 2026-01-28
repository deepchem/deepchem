import numpy as np
from deepchem.utils.typing import PymatgenStructure
from deepchem.feat import MaterialStructureFeaturizer
from deepchem.feat.graph_data import GraphData

class M3GNetFeaturizer(MaterialStructureFeaturizer):
    """
    Featurizer for M3GNet models.
    
    Computes 3-body interactions (bond angles) in addition to standard graph features.
    
    This featurizer constructs a graph representation where:
    - Nodes represent atoms.
    - Edges represent bonds (pairs within `cutoff`).
    - Triplets represent 3-body interactions (pairs of edges sharing the same source atom, within `threebody_cutoff`).
    
    Examples
    --------
    >>> import deepchem as dc
    >>> from deepchem.feat.material_featurizers.m3gnet_featurizer import M3GNetFeaturizer
    >>> import pymatgen.core as mg
    >>> # Create a simple crystal structure (CsCl)
    >>> lattice = mg.Lattice.cubic(4.2)
    >>> structure = mg.Structure(lattice, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    >>> # Initialize featurizer
    >>> featurizer = M3GNetFeaturizer(cutoff=5.0, threebody_cutoff=4.0)
    >>> # Featurize the structure
    >>> graph = featurizer.featurize([structure])[0]
    >>> # Check features
    >>> print("Node features (Z):", graph.node_features.T)
    >>> print("Number of edges:", graph.num_edges)
    >>> print("Number of triplets:", len(graph.threebody_indices))
    """
    
    def __init__(self, cutoff: float = 5.0, threebody_cutoff: float = 4.0):
        """
        Parameters
        ----------
        cutoff: float
            Cutoff radius for the graph (2-body interactions).
        threebody_cutoff: float
            Cutoff radius for 3-body interactions (angles).
        """
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff
        
    def _featurize(self, datapoint: PymatgenStructure, **kwargs) -> GraphData:
        """
        Calculate crystal graph features from pymatgen structure.
        """
        import traceback
        try:
            if 'struct' in kwargs and datapoint is None:
                datapoint = kwargs.get("struct")
                
            # 1. Get Neighbors for Graph (2-body)
            neighbors_list = datapoint.get_all_neighbors(self.cutoff, include_index=True)
            print("neighbours list", neighbors_list)
            # Re-implementation for correct edge tracking with PBC
            all_edges = [] # List of (src, dst, dist, neighbor_obj)
            counter = 0
            neighbors_within_cutoff_per_site = [] # Store indices of edges in 'all_edges' that originate from site i
            
            for i, neighbors in enumerate(neighbors_list):
                site_edges = []
                for n in neighbors:
                    j = n[2]
                    dist = n[1]
                    # edge logic: (src, dst, dist)
                    # Note: M3GNet uses directed graph.
                    all_edges.append((i, j, dist))
                    if dist <= self.threebody_cutoff:
                        site_edges.append((counter, n)) # (edge_index, neighbor)
                    counter += 1
                neighbors_within_cutoff_per_site.append(site_edges)
                
            if not all_edges:
                 # Handle empty graph case if necessary
                 edge_index = np.empty((2, 0), dtype=int)
                 edge_features = np.empty((0, 1), dtype=float)
                 angles = np.empty((0,), dtype=float)
                 threebody_indices = np.empty((0, 2), dtype=int)
            else:
                edge_index = np.array([[e[0], e[1]] for e in all_edges], dtype=int).T
                edge_features = np.array([e[2] for e in all_edges], dtype=float).reshape(-1, 1)
                
                triple_edge_indices = [] # [edge_index_ji, edge_index_ki]
                angles = []
                
                for i, site_edges in enumerate(neighbors_within_cutoff_per_site):
                    # site_edges contains (edge_idx, neighbor_obj) for neighbors <= threebody_cutoff
                    # We want all pairs (j, k)
                    for (e_idx_j, n_j) in site_edges:
                        for (e_idx_k, n_k) in site_edges:
                            if e_idx_j == e_idx_k:
                                 continue
                            
                            # Calculate angle
                            # Vector ij = n_j_site.coords - center.coords 
                            vec_ij = n_j[0].coords - datapoint[i].coords
                            vec_ik = n_k[0].coords - datapoint[i].coords
                            
                            dot = np.dot(vec_ij, vec_ik)
                            norm_curr = n_j[1] * n_k[1]
                            cos_theta = dot / norm_curr
                            cos_theta = np.clip(cos_theta, -1.0, 1.0)
                            angle = np.arccos(cos_theta)
                            
                            angles.append(angle)
                            triple_edge_indices.append([e_idx_j, e_idx_k])

                if triple_edge_indices:
                    threebody_indices = np.array(triple_edge_indices, dtype=int).T
                    angles = np.array(angles, dtype=float)
                else:
                     threebody_indices = np.empty((0, 2), dtype=int)
                     angles = np.empty((0,), dtype=float)

            # 2. Node Features (Atomic Numbers)
            node_features = np.array([site.specie.number for site in datapoint], dtype=int).reshape(-1, 1)
            
            graph = GraphData(
                node_features=node_features,
                edge_index=edge_index,
                edge_features=edge_features,
                angle=angles, 
                threebody_indices=threebody_indices
            )
            return graph
        except Exception:
            traceback.print_exc()
            raise

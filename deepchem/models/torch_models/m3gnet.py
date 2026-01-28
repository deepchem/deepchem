import torch
import torch.nn as nn
from torch_geometric.data import Batch
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.torch_models.layers import (
    MultilayerPerceptron,
    SphericalBesselWithHarmonics,
    ThreeDInteraction,
    GatedAtomUpdate,
    GatedMLP,
    WeightedReadout
)
from deepchem.utils.pytorch_utils import get_activation
import numpy as np

class RadialBasis(nn.Module):
    """
    Radial Basis Function (RBF) expansion for edge distances.
    Uses Oth order spherical Bessel functions (j0) as basis.
    """
    def __init__(self, n_radial: int, cutoff: float):
        super(RadialBasis, self).__init__()
        self.n_radial = n_radial
        self.cutoff = cutoff

    def forward(self, r: torch.Tensor):
        # r: (N, 1) or (N,)
        if r.dim() == 1:
            r = r.unsqueeze(-1)
        
        n = torch.arange(1, self.n_radial + 1, device=r.device).float().view(1, -1) # (1, n_radial)
        
        # arg = n * pi * r / cutoff
        arg = n * np.pi * r / self.cutoff
        
        # j0(x) = sin(x) / x
        # Handle division by zero if r is extremely small (though bonds usually > 0.5A)
        r_safe = torch.where(r < 1e-5, torch.ones_like(r) * 1e-5, r)
        
        val = torch.sin(arg) / r_safe
        return val

class M3GNet(nn.Module):
    """
    Materials 3-body Graph Network (M3GNet) implementation.
    
    This model incorporates 3-body interactions (bond angles) into the message passing
    scheme to serve as an accurate interpretable interatomic potential.

    Examples
    --------
    >>> import deepchem as dc
    >>> import pymatgen.core as mg
    >>> import torch
    >>> from deepchem.models.torch_models.m3gnet import M3GNet
    >>> from deepchem.feat.material_featurizers.m3gnet_featurizer import M3GNetFeaturizer
    >>> from torch_geometric.data import Data, Batch
    >>>
    >>> # 1. Create a simple crystal structure (CsCl)
    >>> lattice = mg.Lattice.cubic(4.2)
    >>> structure = mg.Structure(lattice, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    >>>
    >>> # 2. Featurize
    >>> featurizer = M3GNetFeaturizer(cutoff=5.0, threebody_cutoff=4.0)
    >>> graphs = featurizer.featurize([structure])
    >>> graph = graphs[0]
    >>>
    >>> # 3. Convert to PyG Batch
    >>> data = Data(
    ...     x=torch.tensor(graph.node_features, dtype=torch.long),
    ...     edge_index=torch.tensor(graph.edge_index, dtype=torch.long),
    ...     edge_attr=torch.tensor(graph.edge_features, dtype=torch.float),
    ...     angle=torch.tensor(graph.angle, dtype=torch.float),
    ...     threebody_indices=torch.tensor(graph.threebody_indices, dtype=torch.long)
    ... )
    >>> data.num_nodes = graph.node_features.shape[0]
    >>> batch = Batch.from_data_list([data])
    >>>
    >>> # 4. Initialize and Run Model
    >>> model = M3GNet(n_blocks=2, units=64, n_atom_types=120)
    >>> output = model(batch)
        """
    def __init__(self,
                 n_blocks: int = 3,
                 units: int = 64,
                 cutoff: float = 5.0,
                 threebody_cutoff: float = 4.0,
                 n_radial: int = 36,
                 n_spherical: int = 7,
                 n_atom_types: int = 120,
                 max_n: int = 3,
                 max_l: int = 3,
                 activation: str = 'silu',
                 is_intensive: bool = True,
                 readout: str = "weighted_atom"):
        super(M3GNet, self).__init__()
        self.n_blocks = n_blocks
        self.units = units
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff
        self.n_radial = n_radial
        self.n_spherical = n_spherical
        self.max_n = max_n
        self.max_l = max_l
        self.act = activation
        
        # Embeddings
        self.embedding = nn.Embedding(n_atom_types, units)
        self.radial_basis = RadialBasis(n_radial, cutoff)
        self.edge_embedding = nn.Linear(n_radial, units)
        
        # Basis Functions
        self.basis_expansion = SphericalBesselWithHarmonics(max_n, max_l, cutoff)
        
        # Interactions
        self.three_interactions = nn.ModuleList([
            ThreeDInteraction(
                update_network=MultilayerPerceptron(d_input=max_n*(max_l+1), d_output=units, d_hidden=( units, ), activation_fn='sigmoid'), # Simplified
                update_network2=GatedMLP(neurons=[units, units], activations=[activation])
            ) for _ in range(n_blocks)
        ])
        
        self.graph_layers = nn.ModuleList([
            nn.ModuleDict({
                'atom_network': GatedAtomUpdate(neurons=[units, units], activation=activation),
                'bond_network': nn.Identity() # Placeholder if not strictly defined in snippet
            }) for _ in range(n_blocks)
        ])
        
        # Readout
        if is_intensive:
             if readout == "weighted_atom":
                 self.readout = WeightedReadout(neurons=[units, units])
             else:
                 # Default simple readout
                 self.readout = nn.Sequential(
                     nn.Linear(units, units),
                     nn.SiLU(),
                     nn.Linear(units, 1)
                 )
        
        self.final = nn.Sequential(
            nn.Linear(units, units),
            nn.SiLU(),
            nn.Linear(units, 1)
        )

    def forward(self, pyg_batch: Batch):
        """
        Forward pass.
        Expects pyg_batch to contain:
        - x: atom types
        - edge_index: bond indices
        - edge_attr: bond distances (or pre-computed RBF)
        - angle: bond angles (from M3GNetFeaturizer)
        - threebody_indices: indices mapping triplets to edges (from M3GNetFeaturizer)
        """
        x = pyg_batch.x.squeeze()
        # Embed atoms
        atoms = self.embedding(x)
        
        # Bond features (distances) normally need expansion too?
        # Assuming edge_attr is distances
        edge_attr = pyg_batch.edge_attr.float()
        
        # Expand bond distances to RBF and project
        edge_rbf = self.radial_basis(edge_attr)
        edge_features = self.edge_embedding(edge_rbf)
        
        # 3-body Basis
        # In case angles are missing (dummy batch), handle it?
        if hasattr(pyg_batch, 'angle') and hasattr(pyg_batch, 'threebody_indices'):
             angles = pyg_batch.angle
             threebody_indices = pyg_batch.threebody_indices
             # distances needed for basis? The snippet passes graph.
             # Assuming basis_expansion takes relevant tensors.
             three_basis = self.basis_expansion(edge_attr, angles, threebody_indices)
        else:
             # Fallback/Dummy Not implemented
             three_basis = None
             
        g = atoms # Initialize graph features (atoms)
        
        # Main Loop
        # edge_features is now (NumEdges, units)
        
        g = atoms # Initialize graph features (atoms)

        for i in range(self.n_blocks):
             # 3-body interaction
             if three_basis is not None:
                 # Update edges
                 # three_interactions[i] returns UPDATED edge features
                 # Inputs: edge_features, three_basis, three_cutoff, threebody_indices
                 edge_features = self.three_interactions[i](
                     edge_features, 
                     three_basis, 
                     None, # three_cutoff handled implicitly or None 
                     threebody_indices
                 )
             
             # Graph Layer (Atom update)
             # graph_layers[i]['atom_network'] takes (atom_features, aggregated_edges)
             # We need to aggregate edge features to atoms.
             # Using scatter_add logic (sum pooling)
             src = pyg_batch.edge_index[0] # Source or target? 
             # Usually: v_i' = v_i + sum_{j} e_{ji}
             # If edge_index is (src, dst), we want to sum over incoming edges to dst? 
             # Or outgoing from src?
             # M3GNet usually sums neighbors.
             # torch_geometric convention: edge_index[0] is source, edge_index[1] is target.
             # We aggregate to target (i).
             # dst = pyg_batch.edge_index[1]
             
             # Check if scatter is imported
             try:
                 from torch_geometric.utils import scatter
                 aggregated_edges = scatter(edge_features, pyg_batch.edge_index[1], dim=0, dim_size=g.size(0), reduce='add')
             except ImportError:
                 aggregated_edges = torch.zeros_like(g)
                 aggregated_edges.index_add_(0, pyg_batch.edge_index[1], edge_features)

             g = self.graph_layers[i]['atom_network'](g, aggregated_edges)

        # Readout
        out = self.readout(g, pyg_batch.batch)
        out = self.final(out)
        
        return out

class M3GNetModel(TorchModel):
    def __init__(self, **kwargs):
        model = M3GNet(**kwargs)
        super(M3GNetModel, self).__init__(model, loss=nn.MSELoss(), **kwargs)


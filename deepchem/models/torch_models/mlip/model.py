from deepchem.models import TorchModel
from deepchem.models.losses import L2Loss
import torch.nn as nn
from .layers import RadialEmbeddingBlock
from .energies import AtomicEnergyHead

class MLIPModel(TorchModel):
    """
    PyTorch implementation of Machine Learning Interatomic Potentials (MLIP).
    
    This class wraps the MACE architecture components for use in DeepChem.
    Currently supports Radial Embeddings and Atomic Energy readout.
    """
    def __init__(self, dataset_info, r_max=5.0, num_bessel=8, **kwargs):
        # Build the underlying PyTorch Module
        module = self._build_module(dataset_info, r_max, num_bessel)
        
        super(MLIPModel, self).__init__(
            model=module,
            loss=L2Loss(),
            **kwargs
        )

    def _build_module(self, dataset_info, r_max, num_bessel):
        """Constructs the internal nn.Module."""
        class _Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.radial = RadialEmbeddingBlock(r_max, num_bessel)
                self.energy_head = AtomicEnergyHead(dataset_info, "average")
            
            def forward(self, inputs):
                # MVP Forward pass logic
                # 1. Radial features
                # 2. Atomic energies
                return self.energy_head(inputs)
        
        return _Wrapper()
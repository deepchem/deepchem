import torch
from typing import Optional, Union, Dict

class AtomicEnergyHead(torch.nn.Module):
    """
    Handles atomic energy readout for MLIP models.
    Supports average energy, zero initialization, or custom dictionaries.
    """
    def __init__(self, 
                 dataset_info, 
                 input_method: str = "average"):
        super().__init__()
        self.energies = self._get_atomic_energies(dataset_info, input_method)
        # Register as buffer so it moves to GPU automatically
        self.register_buffer("atomic_energies", self.energies)

    def _get_atomic_energies(self, dataset_info, method) -> torch.Tensor:
        # Extract atomic map from dataset_info object (DeepChem standard)
        if hasattr(dataset_info, 'atomic_energies_map'):
            mapping = dataset_info.atomic_energies_map
        else:
            # Fallback if it's just a dict
            mapping = dataset_info
            
        zs = sorted(mapping.keys())
        
        if method == "average":
            return torch.tensor([mapping[z] for z in zs], dtype=torch.float32)
        elif method == "zero":
            return torch.zeros(len(zs), dtype=torch.float32)
        else:
            raise ValueError(f"Unknown energy method: {method}")

    def forward(self, x):
        # Placeholder for returning fixed energies
        return self.atomic_energies
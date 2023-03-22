from typing import List
from dataclasses import dataclass

try:
    import torch
except ModuleNotFoundError:
    pass


@dataclass
class GroverBatchMolGraph:
    smiles_batch: List[str]
    f_atoms: torch.FloatTensor
    f_bonds: torch.FloatTensor
    fg_labels: torch.FloatTensor
    additional_feats: torch.FloatTensor
    a2b: torch.LongTensor
    b2a: torch.LongTensor
    b2revb: torch.LongTensor
    a2a: torch.LongTensor
    a_scope: torch.LongTensor
    b_scope: torch.LongTensor

    def get_components(self):
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope, self.a2a

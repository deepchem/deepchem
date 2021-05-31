"""
Huggingface/transformers RoBERTa model for sequence-based property prediction.
"""

import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import RobertaForMaskedLM


class ChemBERTa(nn.Module):
    def __init__(self):
        try:
            import transformers
        except:
            raise ImportError('This class requires transformers.')

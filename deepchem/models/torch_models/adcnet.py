import torch
import torch.nn as nn
from transformers import EsmTokenizer, EsmModel
from deepchem.models.torch_models.chemberta import Chemberta

class ADCNet(nn.Module):
    """
    ADCNet
    
    ADCNet is a sophisticated model engineered to seamlessly process both protein 
    sequences and chemical structures for integrated biological and chemical analysis. 
    The model begins by embedding antibody and antigen protein sequences using ESM-2, 
    a specialized tool designed for high-quality protein sequence representation. 
    Simultaneously, it embeds payload and linker SMILES strings—textual representations 
    of chemical structures—using ChemBERTa, a transformer-based model tailored for chemical 
    data. These four embeddings, two from the protein sequences and two from the SMILES
    strings are then concatenated into a single, unified vector. This combined vector is 
    subsequently processed through a 3-layer multilayer perceptron (MLP), enabling the
    model to perform advanced analysis and predictions by leveraging the integrated 
    features of both biological and chemical inputs. This streamlined architecture empowers
    ADCNet to effectively bridge diverse data types, making it a powerful tool for applications 
    requiring the fusion of protein and chemical information.
    
    Args:
      seq_model_name: HuggingFace name for ESM-2 (e.g. 'facebook/esm2_t6_8M_UR50D')
      chemberta_task: one of ['mlm','mtr','regression','classification']
      chemberta_tokenizer: HF path for the ChemBERTa tokenizer
      hidden_dim: width of the MLP hidden layer
      output_dim: final output dimension (e.g. number of regression targets)
    """
    def __init__(self,
                 seq_model_name: str = 'facebook/esm2_t6_8M_UR50D',
                 chemberta_task: str = 'regression',
                 chemberta_tokenizer: str = 'seyonec/PubChem10M_SMILES_BPE_60k',
                 hidden_dim: int = 128,
                 output_dim: int = 1):
        super(ADCNet, self).__init__()
        # ---- ESM-2 embedder ----
        self.esm_tokenizer = EsmTokenizer.from_pretrained(seq_model_name)
        self.esm_model     = EsmModel.from_pretrained(seq_model_name)
        
        self.esm_model.eval()
        for p in self.esm_model.parameters():
            p.requires_grad = False

        # ChemBERTa embedder
        # DeepChem’s Chemberta wraps a HF RoBERTa under the hood
        self.chemberta = Chemberta(
            task=chemberta_task,
            tokenizer_path=chemberta_tokenizer
        )
        
        self.chemberta.model.eval()
        for p in self.chemberta.model.parameters():
            p.requires_grad = False

        # ---- downstream MLP ----
        # ESM2 hidden size + ChemBERTa hidden size =  512 (for esm2_t6) + 768 = 1280
        seq_emb_dim = self.esm_model.config.hidden_size
        chem_emb_dim = self.chemberta.model.config.hidden_size
        total_dim = seq_emb_dim + chem_emb_dim * 2  # for payload & linker

        self.fc1 = nn.Linear(total_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def _embed_sequence(self, seq_batch: torch.Tensor) -> torch.Tensor:
        """
        Tokenize protein sequences and return the CLS (first token) embedding.
        seq_batch: list of strings, length B
        returns: (B, hidden_size)
        """
        tokens = self.esm_tokenizer(seq_batch,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True)
        tokens = {k: v.to(self.esm_model.device) for k, v in tokens.items()}
        with torch.no_grad():
            out = self.esm_model(**tokens).last_hidden_state
        # take first token ([CLS]) for each sequence
        return out[:, 0, :]

    def _embed_smiles(self, smiles_batch: torch.Tensor) -> torch.Tensor:
        """
        Use DeepChem’s Chemberta to embed SMILES.
        smiles_batch: list of strings, length B
        returns: (B, hidden_size)
        """
        # HuggingFaceModel.predict returns numpy, so we’ll re-tensorize
        preds = self.chemberta.predict(smiles_batch)
        # preds is an (N, hidden_size) array for embeddings (task='mlm' returns logits; for 'regression' it's final layer)
        return torch.tensor(preds, device=self.fc1.weight.device, dtype=torch.float32)

    def forward(self,
                seqs: torch.Tensor,
                smiles_payload: torch.Tensor,
                smiles_linker: torch.Tensor) -> torch.Tensor:
        """
        Args:
          seqs: list of protein sequences (antibody+antigen concatenated, or pass separately)
          smiles_payload: list of payload SMILES
          smiles_linker: list of linker SMILES
        """
        # get embeddings
        seq_emb    = self._embed_sequence(seqs)             # (B, seq_emb_dim)
        ply_emb    = self._embed_smiles(smiles_payload)     # (B, chem_emb_dim)
        link_emb   = self._embed_smiles(smiles_linker)      # (B, chem_emb_dim)

        # concat
        x = torch.cat([seq_emb, ply_emb, link_emb], dim=1)   # (B, total_dim)

        # downstream MLP
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = self.fc3(x)
        return out

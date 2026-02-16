import numpy as np
import deepchem as dc
from deepchem.models.torch_models.chemberta import Chemberta

# -----------------------------
# 1. Small Toy Dataset
# -----------------------------
# Two SMILES molecules
smiles = ["CCO", "CCN"]   # ethanol, ethylamine
labels = np.array([1, 0])  # dummy binary labels

# IMPORTANT:
# We pass RAW SMILES strings.
# HuggingFaceModel will tokenize internally.
dataset = dc.data.NumpyDataset(X=np.array(smiles), y=labels)

# -----------------------------
# 2. Initialize ChemBERTa Model
# -----------------------------
model = Chemberta(
    model_name="seyonec/ChemBERTa-zinc-base-v1",
    num_labels=2,
    task="classification"
)

# -----------------------------
# 3. Train (Tiny demo)
# -----------------------------
model.fit(dataset, nb_epoch=1)

print("ChemBERTa training complete!")

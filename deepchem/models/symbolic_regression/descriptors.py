import deepchem as dc
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import numpy as np

def compute_descriptors(smiles_list):
    descriptors = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        desc = [
            Descriptors.MolLogP(mol),
            Descriptors.MolWt(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
        ]
        descriptors.append(desc)
    return np.array(descriptors)

tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='ECFP')
train, valid, test = datasets

X_train = compute_descriptors(train.ids)
y_train = train.y.flatten()

print(f"Shape: {X_train.shape}")
print(f"First molecule descriptors: {X_train[0]}")
print(f"Feature names: LogP, MolWt, HBD, HBA, TPSA, RotBonds")
# This example shows how to use Pandas to load data directly
# without using a CSVLoader object.
# It also demonstrates how to safely handle invalid SMILES strings
# when converting data into a DeepChem Dataset.

import pandas as pd
import deepchem as dc
from rdkit import Chem

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("example.csv")
print("Original data loaded as DataFrame:")
print(df)

# Initialize the featurizer
featurizer = dc.feat.CircularFingerprint(size=16)

# Lists to store valid molecules and corresponding indices
mols = []
valid_indices = []

# Convert SMILES strings to RDKit molecule objects
# Chem.MolFromSmiles can return None for invalid SMILES,
# so we explicitly filter those out to avoid featurization errors.
for idx, smiles in enumerate(df["smiles"]):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mols.append(mol)
        valid_indices.append(idx)

# Featurize only valid molecules
features = featurizer.featurize(mols)

# Select labels and IDs corresponding to valid molecules
labels = df.loc[valid_indices, "log-solubility"]
ids = df.loc[valid_indices, "Compound ID"]

# Create a DeepChem Dataset
dataset = dc.data.NumpyDataset(
    X=features,
    y=labels,
    ids=ids
)

print("Data converted into DeepChem Dataset")
print(dataset)

# Convert the DeepChem Dataset back into a pandas DataFrame
# This is useful for inspection or further processing with pandas
converted_df = dataset.to_dataframe()
print("Data converted back into DataFrame:")
print(converted_df)

# Example demonstrating how to load a CSV files using pandas and convert it into a DeepChem Dataset for further processing .
# Import Required Libraries

import pandas as pd
import deepchem as dc
from rdkit import Chem

#Load the CSV file into a Pandas Data Frame
df = pd.read_csv("example.csv")
print("Original data loaded as DataFrame:")
print(df)

#Initialize a circular fingerprint featurizer
featurizer = dc.feat.CircularFingerprint(size=16)

#Convert SMILES string into RDKit molecules objects
mols = [Chem.MolFromSmiles(smiles) for smiles in df["smiles"]]

#Generate molecular features from the molecules
features = featurizer.featurize(mols)

#Create a DeepChem dataset from features and labels
dataset = dc.data.NumpyDataset(
    X=features, y=df["log-solubility"], ids=df["Compound ID"])

print("Data converted into DeepChem Dataset")
print(dataset)

# Convert the DeepChem dataset back to a pandas DataFrame
converted_df = dataset.to_dataframe()
print("Data converted back into DataFrame:")
print(converted_df)

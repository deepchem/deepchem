# This example shows how to use Pandas to load data directly
# without using a CSVLoader object. This may be useful if you
# want the flexibility of processing your data with Pandas
# directly.
import pandas as pd
import deepchem as dc

df = pd.read_csv("example.csv")
print("Original data loaded as DataFrame:")
print(df)

featurizer = dc.feat.CircularFingerprint(size=16)
features = featurizer.featurize(df["smiles"])
dataset = dc.data.NumpyDataset(X=features, y=df["log-solubility"], ids=df["Compound ID"])

print("Data converted into DeepChem Dataset")
print(dataset)

# Now let's convert from a dataset back to a pandas dataframe
converted_df = dataset.to_dataframe()
print("Data converted back into DataFrame:")
print(converted_df)

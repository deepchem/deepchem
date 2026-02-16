from deepchem.models.torch_models.dnabert import DNABERTModel
import deepchem as dc
import numpy as np

# Example 1: Binary Classification
print("=== Binary Classification Example ===")
sequences = ["ACGTACGTACGT", "ATCGATCGATCG", "GGCCGGCCGGCC"]
labels = np.array([1, 0, 1])

dataset = dc.data.NumpyDataset(X=np.array(sequences), y=labels)

model = DNABERTModel(
    task='classification',
    model_name="zhihan1996/DNA_bert_6",
    n_tasks=1
)

model.fit(dataset, nb_epoch=2)
predictions = model.predict(dataset)
print(f"Predictions: {predictions}")

# Example 2: Regression
print("\n=== Regression Example ===")
binding_scores = np.array([[3.2], [5.7], [4.1]])
dataset_reg = dc.data.NumpyDataset(X=np.array(sequences), y=binding_scores)

model_reg = DNABERTModel(
    task='regression',
    model_name="zhihan1996/DNA_bert_6",
    n_tasks=1
)

model_reg.fit(dataset_reg, nb_epoch=2)
predictions_reg = model_reg.predict(dataset_reg)
print(f"Regression predictions: {predictions_reg}")

print("\n All examples completed!")

# ChemBERTa-3 Style (No pre-featurization)
# import deepchem as dc
# from deepchem.models.torch_models.chemberta import Chemberta

# # 1. Raw SMILES in the dataset
# smiles = ["C1=CC=CC=C1", "CC(=O)O"]
# labels = [1, 0]
# dataset = dc.data.NumpyDataset(X=smiles, y=labels)

# # 2. The Model takes the tokenizer name
# model = Chemberta(
#     model_name="deepchem/chemberta-770M-MTR",
#     task="classification"
# )

# # 3. fit() does the tokenization INTERNALLY
# model.fit(dataset)
# print("Success..")
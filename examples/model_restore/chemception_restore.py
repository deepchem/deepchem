import deepchem as dc
import rdkit.Chem as Chem

model = dc.models.ChemCeption(
    img_spec="engd", n_tasks=1, model_dir="./model", mode="regression")
model.restore()

smiles = "CCCCC"
featurizer = dc.feat.SmilesToImage(img_spec="engd", img_size=80, res=0.5)
dataset = dc.data.NumpyDataset(
    featurizer.featurize([Chem.MolFromSmiles(smiles)]))
prediction = model.predict(dataset)
print("smiles: %s" % smiles)
print("prediction: %s" % str(prediction))

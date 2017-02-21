import numpy as np

from deepchem.data import DiskDataset
from deepchem.models.autoencoder_models.autoencoder import TensorflowMoleculeEncoder, TensorflowMoleculeDecoder
from deepchem.trans.transformers import OneHotTransformer
from deepchem.metrics import Metric
import deepchem.metrics as metrics

data_dir = "/home/leswing/Documents/data_sets/keras-molecule"
charset = [' ', '#', ')', '(', '+', '-', '/', '1', '3', '2', '5', '4', '7', '6', '8', '=', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'S', '[', ']', '\\', 'c', 'l', 'o', 'n', 'p', 's', 'r']

tf_enc = TensorflowMoleculeEncoder(model_dir=data_dir, charset=charset)

smiles = ["Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC(=O)N1CN(C(C)=O)[C@@H](O)[C@@H]1O"]
smiles = [[x] for x in smiles]
smiles = np.array(smiles)

dataset = DiskDataset.from_numpy(smiles, smiles, data_dir=data_dir)
x_transformer = OneHotTransformer(transform_X=True, transform_y=False, dataset=dataset)
y_transformer = OneHotTransformer(transform_X=False, transform_y=True, dataset=dataset)
dataset = x_transformer.transform(dataset)

#accuracy_metric = Metric(metrics.accuracy_score, verbose=True)
#tf_enc.evaluate(dataset, [accuracy_metric], [x_transformer, y_transformer])
prediction = tf_enc.predict_on_batch(dataset.X)

tf_de = TensorflowMoleculeDecoder(model_dir=data_dir, charset=charset)
one_hot_decoded = tf_de.predict_on_batch(prediction)
print(x_transformer.untransform(one_hot_decoded))


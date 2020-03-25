from rdkit import Chem

from deepchem.data import NumpyDataset
from deepchem.models.autoencoder_models.autoencoder import TensorflowMoleculeEncoder, TensorflowMoleculeDecoder
from deepchem.feat.one_hot import OneHotFeaturizer, zinc_charset

tf_enc = TensorflowMoleculeEncoder.zinc_encoder()

smiles = ["Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC(=O)N1CN(C(C)=O)[C@@H](O)[C@@H]1O"]
mols = [Chem.MolFromSmiles(x) for x in smiles]
featurizer = OneHotFeaturizer(zinc_charset)
features = featurizer.featurize(mols)

dataset = NumpyDataset(features, features)
prediction = tf_enc.predict_on_batch(dataset.X)

tf_de = TensorflowMoleculeDecoder.zinc_decoder()
one_hot_decoded = tf_de.predict_on_batch(prediction)
print(featurizer.untransform(one_hot_decoded))

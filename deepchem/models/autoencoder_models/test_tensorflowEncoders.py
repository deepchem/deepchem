import numpy as np
from unittest import TestCase

from nose.tools import assert_equals

import deepchem as dc
from data import DiskDataset
from models.autoencoder_models.autoencoder import TensorflowMoleculeEncoder, TensorflowMoleculeDecoder
from deepchem.feat.one_hot import OneHotFeaturizer, zinc_charset
from rdkit import Chem


class TestTensorflowEncoders(TestCase):
  def test_fit(self):
    data_dir = "/home/leswing/Documents/data_sets/keras-molecule"

    tf_enc = TensorflowMoleculeEncoder(model_dir=data_dir)

    smiles = ["Cn1cnc2c1c(=O)n(C)c(=O)n2C",
              "O=C(O)[C@@H]1/C(=C/CO)O[C@@H]2CC(=O)N21",
              "Cn1c2nncnc2c(=O)n(C)c1=O",
              "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
              "NC(=O)c1ncc[nH]c1=O",
              "O=C1OCc2c1[nH]c(=O)[nH]c2=O",
              "Cn1c(N)c(N)c(=O)n(C)c1=O",
              "CNc1nc2c([nH]1)c(=O)[nH]c(=O)n2C",
              "CC(=O)N1CN(C(C)=O)[C@@H](O)[C@@H]1O",
              "CC(=O)N1CN(C(C)=O)[C@H](O)[C@H]1O",
              "Cc1[nH]c(=O)[nH]c(=O)c1CO",
              "O=C1NCCCc2c1no[n+]2[O-]",
              "Cc1nc(C(N)=O)c(N)n1CCO",
              "O=c1[nH]cc(N2CCOCC2)c(=O)[nH]1"]

    featurizer = dc.feat.one_hot.OneHotFeaturizer(zinc_charset, 120)
    mols = [Chem.MolFromSmiles(x) for x in smiles]
    features = featurizer.featurize(mols)

    dataset = DiskDataset.from_numpy(features, features)
    #x_transformer = OneHotTransformer(transform_X=True, transform_y=False, dataset=dataset)
    #dataset = x_transformer.transform(dataset)

    prediction = tf_enc.predict_on_batch(dataset.X)

    tf_de = TensorflowMoleculeDecoder(model_dir=data_dir)
    one_hot_decoded = tf_de.predict_on_batch(prediction)
    decoded_smiles = featurizer.untransform(one_hot_decoded)
    assert_equals(len(decoded_smiles), len(smiles))
    for i in range(len(smiles)):
      print(decoded_smiles[i], smiles[i])

import tempfile
import os
import deepchem as dc
import numpy as np


def test_flattening_with_csv_load_withtask():
    fin = tempfile.NamedTemporaryFile(mode='w', delete=False)
    fin.write("smiles,endpoint\nc1ccccc1,1")
    fin.close()
    loader = dc.data.CSVLoader(
        ["endpoint"],
        feature_field="smiles",
        featurizer=dc.feat.ConvMolFeaturizer(per_atom_fragmentation=True))
    frag_dataset = loader.create_dataset(fin.name)
    transformer = dc.trans.FlatteningTransformer(dataset=frag_dataset)
    frag_dataset = transformer.transform(frag_dataset)
    assert len(frag_dataset) == 6
    assert np.shape(frag_dataset.y) == (6, 1
                                       )  # y should be expanded up to X shape
    assert np.shape(frag_dataset.w) == (6, 1
                                       )  # w should be expanded up to X shape


def test_flattening_with_csv_load_notask():
    fin = tempfile.NamedTemporaryFile(mode='w', delete=False)
    fin.write("smiles,endpoint\nc1ccccc1,1")
    fin.close()
    loader = dc.data.CSVLoader(
        [],
        feature_field="smiles",
        featurizer=dc.feat.ConvMolFeaturizer(per_atom_fragmentation=True))
    frag_dataset = loader.create_dataset(fin.name)
    transformer = dc.trans.FlatteningTransformer(dataset=frag_dataset)
    frag_dataset = transformer.transform(frag_dataset)
    assert len(frag_dataset) == 6


def test_flattening_with_sdf_load_withtask():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    featurizer = dc.feat.ConvMolFeaturizer(per_atom_fragmentation=True)
    loader = dc.data.SDFLoader(["LogP(RRCK)"],
                               featurizer=featurizer,
                               sanitize=True)
    dataset = loader.create_dataset(
        os.path.join(cur_dir, "membrane_permeability.sdf"))
    transformer = dc.trans.FlatteningTransformer(dataset=dataset)
    frag_dataset = transformer.transform(dataset)
    assert len(frag_dataset) == 98
    assert np.shape(frag_dataset.y) == (98, 1
                                       )  # y should be expanded up to X shape
    assert np.shape(frag_dataset.w) == (98, 1
                                       )  # w should be expanded up to X shape


def test_flattening_with_sdf_load_notask():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    featurizer = dc.feat.ConvMolFeaturizer(per_atom_fragmentation=True)
    loader = dc.data.SDFLoader([], featurizer=featurizer, sanitize=True)
    dataset = loader.create_dataset(
        os.path.join(cur_dir, "membrane_permeability.sdf"))
    transformer = dc.trans.FlatteningTransformer(dataset=dataset)
    frag_dataset = transformer.transform(dataset)
    assert len(frag_dataset) == 98

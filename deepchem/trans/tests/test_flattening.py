import tempfile
import os
import numpy as np
import deepchem as dc


def test_flattening_with_csv_load():
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


def test_flattening_with_sdf_load():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  featurizer = dc.feat.ConvMolFeaturizer(per_atom_fragmentation=True)
  loader = dc.data.SDFLoader([], featurizer=featurizer, sanitize=True)
  dataset = loader.create_dataset(
      os.path.join(current_dir, "membrane_permeability.sdf"))
  transformer = dc.trans.FlatteningTransformer(dataset=dataset)
  frag_dataset = transformer.transform(dataset)
  assert len(frag_dataset) == 96

import os
import tempfile
import deepchem as dc


def test_load_singleton_csv():
  fin = tempfile.NamedTemporaryFile(mode='w', delete=False)
  fin.write("smiles,endpoint\nc1ccccc1,1")
  fin.close()
  featurizer = dc.feat.CircularFingerprint(size=1024)
  tasks = ["endpoint"]
  loader = dc.data.CSVLoader(
      tasks=tasks, feature_field="smiles", featurizer=featurizer)

  X = loader.create_dataset(fin.name)
  assert len(X) == 1
  os.remove(fin.name)


def test_singleton_csv_load_with_per_atom_fragmentation():
  """Test a case where special form of  dataaset is created from csv:
   dataset of fragments of molecules  for subsequent model interpretation """
  fin = tempfile.NamedTemporaryFile(mode='w', delete=False)
  fin.write("smiles,endpoint\nc1ccccc1,1")
  fin.close()
  featurizer = dc.feat.ConvMolFeaturizer(per_atom_fragmentation=True)
  tasks = ["endpoint"]
  loader = dc.data.CSVLoader(
      tasks=tasks, feature_field="smiles", featurizer=featurizer)

  X = loader.create_dataset(fin.name)
  assert len(X) == 6
  os.remove(fin.name)

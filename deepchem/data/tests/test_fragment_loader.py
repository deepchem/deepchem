import os
import tempfile
import deepchem as dc


def test_singleton_csv_fragment_load_with_per_atom_fragmentation():
  """Test a case where special form of  dataaset is created from csv:
   dataset of fragments of molecules  for subsequent model interpretation """
  with tempfile.NamedTemporaryFile(mode='w', delete=False) as fin:
    fin.write("smiles,endpoint\nc1ccccc1,1")
  featurizer = dc.feat.ConvMolFeaturizer(per_atom_fragmentation=True)
  tasks = ["endpoint"]
  loader = dc.data.CSVFragmentLoader(
      tasks=tasks, feature_field="smiles", featurizer=featurizer)
  X = loader.create_dataset(fin.name)
  assert len(X) == 6
  os.remove(fin.name)


def test_sdf_fragment_load_with_per_atom_fragmentation():
  """Test a case where special form of  dataaset is created from SDF:
    dataset of fragments of molecules  for subsequent model interpretation """
  current_dir = os.path.dirname(os.path.realpath(__file__))
  featurizer = dc.feat.ConvMolFeaturizer(per_atom_fragmentation=True)
  loader = dc.data.SDFFragmentLoader(
      ["LogP(RRCK)"], featurizer=featurizer, sanitize=True)
  dataset = loader.create_dataset(
      os.path.join(current_dir, "membrane_permeability.sdf"))
  assert len(dataset) == 98

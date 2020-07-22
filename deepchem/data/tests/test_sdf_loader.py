import os
import deepchem as dc


def test_sdf_load():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  featurizer = dc.feat.CircularFingerprint(size=16)
  loader = dc.data.SDFLoader(
      ["LogP(RRCK)"], featurizer=featurizer, sanitize=True)
  dataset = loader.create_dataset(
      os.path.join(current_dir, "membrane_permeability.sdf"))
  assert len(dataset) == 2


def test_singleton_sdf_load():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  featurizer = dc.feat.CircularFingerprint(size=16)
  loader = dc.data.SDFLoader(
      ["LogP(RRCK)"], featurizer=featurizer, sanitize=True)
  dataset = loader.create_dataset(os.path.join(current_dir, "singleton.sdf"))
  assert len(dataset) == 1

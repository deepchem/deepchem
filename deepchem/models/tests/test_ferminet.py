"""
Test for Ferminet Model.
"""

import pytest
import numpy as np
from deepchem.models.torch_models.ferminet import Ferminet
from deepchem.models.torch_models.ferminet import FerminetModel

@pytest.mark.torch
def test_FerminetModel():
  # Test for the init function of FerminetModel class
  FH_molecule = [['F',[0, 0, 0]],['H',[0, 0.5, 0.5]]]
  # Testing ionic initialization
  model = FerminetModel(FH_molecule, spin=1, ion_charge=-1)
  assert (model.electron_no == np.array([[10],[1]])).all()
  # Testing spin correct or not
  with pytest.raises(ValueError):
    FerminetModel(FH_molecule, spin=0, ion_charge=-1)
  raise IndexError

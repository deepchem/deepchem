"""
Runs example notebooks to ensure code changes haven't broken notebooks.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "LGPL"

import os
import unittest
import subprocess as sp

class TestNotebooks(unittest.TestCase):
  """
  Test example notebooks."
  """
  def setUp(self):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    self.notebook_dir = os.path.join(current_dir, "../../../examples")

  def _test_notebook(self, notebook_location):
    child = sp.Popen(["runipy", "%s" % notebook_location], stdout=sp.PIPE)
    streamdata = child.communicate()[0]
    rc = child.returncode
    assert rc == 0

  def test_solubility_notebook(self):
    """Test solubility notebook."""
    solubility_notebook = os.path.join(
        self.notebook_dir, "solubility.ipynb")
    self._test_notebook(solubility_notebook)

  #def test_protein_ligand_complex_notebook(self):
  #  """Test protein-ligand complex notebook."""
  #  protein_ligand_complex_notebook = os.path.join(
  #      self.notebook_dir, "protein_ligand_complex_notebook.ipynb")
  #  self._test_notebook(protein_ligand_complex_notebook)

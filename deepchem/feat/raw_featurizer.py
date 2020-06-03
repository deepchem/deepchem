#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from deepchem.feat import Featurizer

try:
  from rdkit import Chem
except ImportError:
  pass


class RawFeaturizer(Featurizer):

  def __init__(self, smiles=False):
    self.smiles = smiles

  def _featurize(self, mol):
    if self.smiles:
      return Chem.MolToSmiles(mol)
    else:
      return mol

import numpy as np


def get_molecule_centroid(molecule_xyz):
  """Uses compute centroid and range of 3D coordinents"""
  return np.mean(molecule_xyz, axis=0)


def get_molecule_range(molecule_xyz):
  protein_max = np.max(molecule_xyz, axis=0)
  protein_min = np.min(molecule_xyz, axis=0)
  protein_range = protein_max - protein_min
  return protein_range

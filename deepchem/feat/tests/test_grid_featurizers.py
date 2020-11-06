import os
import unittest
import deepchem as dc


def test_charge_voxelizer():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  protein_file = os.path.join(current_dir, 'data',
                              '3ws9_protein_fixer_rdkit.pdb')
  ligand_file = os.path.join(current_dir, 'data', '3ws9_ligand.sdf')

  cutoff = 4.5
  box_width = 16
  voxel_width = 1.0
  voxelizer = dc.feat.ChargeVoxelizer(
      cutoff=cutoff, box_width=box_width, voxel_width=voxel_width)
  features, failures = voxelizer.featurize([ligand_file], [protein_file])


def test_salt_bridge_voxelizer():
  pass


def test_cation_pi_voxelizer():
  pass


def test_pi_stack_voxelizer():
  pass


def test_hydrogen_bond_counter():
  pass


def test_hydrogen_bond_voxelizer():
  pass

import os
import deepchem as dc


def test_charge_voxelizer():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, 'data',
                                '3ws9_protein_fixer_rdkit.pdb')
    ligand_file = os.path.join(current_dir, 'data', '3ws9_ligand.sdf')

    cutoff = 4.5
    box_width = 20
    voxel_width = 1.0
    voxelizer = dc.feat.ChargeVoxelizer(cutoff=cutoff,
                                        box_width=box_width,
                                        voxel_width=voxel_width)
    features = voxelizer.featurize([(ligand_file, protein_file)])
    assert features.shape == (1, box_width, box_width, box_width, 1)


def test_salt_bridge_voxelizer():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, 'data',
                                '3ws9_protein_fixer_rdkit.pdb')
    ligand_file = os.path.join(current_dir, 'data', '3ws9_ligand.sdf')

    cutoff = 4.5
    box_width = 20
    voxel_width = 1.0
    voxelizer = dc.feat.SaltBridgeVoxelizer(cutoff=cutoff,
                                            box_width=box_width,
                                            voxel_width=voxel_width)
    features = voxelizer.featurize([(ligand_file, protein_file)])
    assert features.shape == (1, box_width, box_width, box_width, 1)


def test_cation_pi_voxelizer():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, 'data',
                                '3ws9_protein_fixer_rdkit.pdb')
    ligand_file = os.path.join(current_dir, 'data', '3ws9_ligand.sdf')

    cutoff = 4.5
    box_width = 20
    voxel_width = 1.0
    voxelizer = dc.feat.CationPiVoxelizer(cutoff=cutoff,
                                          box_width=box_width,
                                          voxel_width=voxel_width)
    features = voxelizer.featurize([(ligand_file, protein_file)])
    assert features.shape == (1, box_width, box_width, box_width, 1)


def test_pi_stack_voxelizer():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, 'data',
                                '3ws9_protein_fixer_rdkit.pdb')
    ligand_file = os.path.join(current_dir, 'data', '3ws9_ligand.sdf')

    cutoff = 4.5
    box_width = 20
    voxel_width = 1.0
    voxelizer = dc.feat.PiStackVoxelizer(cutoff=cutoff,
                                         box_width=box_width,
                                         voxel_width=voxel_width)
    features = voxelizer.featurize([(ligand_file, protein_file)])
    assert features.shape == (1, box_width, box_width, box_width, 2)


# # TODO: This is failing, something about the hydrogen bond counting?
# def test_hydrogen_bond_counter():
#   current_dir = os.path.dirname(os.path.realpath(__file__))
#   protein_file = os.path.join(current_dir, 'data',
#                               '3ws9_protein_fixer_rdkit.pdb')
#   ligand_file = os.path.join(current_dir, 'data', '3ws9_ligand.sdf')
#
#   cutoff = 4.5
#   featurizer = dc.feat.HydrogenBondCounter(cutoff=cutoff)
#   features, failures = featurizer.featurize([ligand_file], [protein_file])
#   # TODO: Add shape test
#
#
# # TODO: This is failing, something about the hydrogen bond counting?
# def test_hydrogen_bond_voxelizer():
#   current_dir = os.path.dirname(os.path.realpath(__file__))
#   protein_file = os.path.join(current_dir, 'data',
#                               '3ws9_protein_fixer_rdkit.pdb')
#   ligand_file = os.path.join(current_dir, 'data', '3ws9_ligand.sdf')
#
#   cutoff = 4.5
#   box_width = 16
#   voxel_width = 1.0
#   voxelizer = dc.feat.HydrogenBondVoxelizer(
#       cutoff=cutoff, box_width=box_width, voxel_width=voxel_width)
#   features, failures = voxelizer.featurize([ligand_file], [protein_file])
#   # TODO: Add shape test

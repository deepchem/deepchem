# This example featurizes a small subset of the PDBBind v2015 protein-ligand core data with the AtomicConvFeaturizer
import deepchem as dc
import logging
from deepchem.molnet.load_function.pdbbind_datasets import get_pdbbind_molecular_complex_files

complex_files = get_pdbbind_molecular_complex_files(subset="core", version="v2015", interactions="protein-ligand", load_binding_pocket=False)
n_featurize = 2
core_subset = complex_files[:n_featurize]

box_width = 48 
voxel_width = 2
voxels_per_edge = box_width/voxel_width
voxelizer = dc.feat.ContactCircularVoxelizer(box_width=box_width, voxel_width=voxel_width)
features, failures = voxelizer.featurize_complexes(
    core_subset, parallelize=False)
print("features.shape")
print(features.shape)
print("%d failures" % len(failures))
assert features.shape == (voxels_per_edge, voxels_per_edge, voxels_per_edge, 8)

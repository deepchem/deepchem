#import os
#from deepchem.utils.save import load_from_disk, save_to_disk
#from deepchem.featurizers.fingerprints import CircularFingerprint
#from deepchem.featurizers.basic import RDKitDescriptors
#from deepchem.featurizers.nnscore import NNScoreComplexFeaturizer
#from deepchem.featurizers.grid_featurizer import GridFeaturizer
#from deepchem.featurizers.featurize import DataLoader
#
#dataset_file = "../../../datasets/pdbbind_full_df.pkl.gz"
#print("About to load dataset form disk.")
#dataset = load_from_disk(dataset_file)
#print("Loaded dataset.")
#
#grid_featurizer = GridFeaturizer(
#    voxel_width=16.0, feature_types="voxel_combined",
#    voxel_feature_types=["ecfp", "splif", "hbond", "pi_stack", "cation_pi",
#    "salt_bridge"], ecfp_power=9, splif_power=9,
#    parallel=True, flatten=True)
#featurizers = [CircularFingerprint(size=1024)]
#featurizers += [grid_featurizer, NNScoreComplexFeaturizer()]
#
##Make a directory in which to store the featurized complexes.
#base_dir = "../../../grid_nnscore_circular_features"
#if not os.path.exists(base_dir):
#    os.makedirs(base_dir)
#data_dir = os.path.join(base_dir, "data")
#if not os.path.exists(data_dir):
#    os.makedirs(data_dir)
#    
#featurized_samples_file = os.path.join(data_dir, "featurized_samples.joblib")
#
#feature_dir = os.path.join(base_dir, "features")
#if not os.path.exists(feature_dir):
#    os.makedirs(feature_dir)
#
#samples_dir = os.path.join(base_dir, "samples")
#if not os.path.exists(samples_dir):
#    os.makedirs(samples_dir)
#
#
#
#featurizers = compound_featurizers + complex_featurizers
#featurizer = DataLoader(tasks=["label"],
#                        smiles_field="smiles",
#                        protein_pdb_field="protein_pdb",
#                        ligand_pdb_field="ligand_pdb",
#                        compound_featurizers=compound_featurizers,
#                        complex_featurizers=complex_featurizers,
#                        id_field="complex_id",
#                        verbose=False)
#from ipyparallel import Client
#c = Client()
#print("c.ids")
#print(c.ids)
#dview = c[:]
#featurized_samples = featurizer.featurize(dataset_file, feature_dir, samples_dir,
#                                          worker_pool=dview, shard_size=1024)
#
#save_to_disk(featurized_samples, featurized_samples_file)

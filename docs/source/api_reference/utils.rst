Utilities
=========
DeepChem has a broad collection of utility functions. Many of these
maybe be of independent interest to users since they deal with some
tricky aspects of processing scientific datatypes.

Data Utilities
--------------

Array Utilities
^^^^^^^^^^^^^^^

.. autofunction:: deepchem.utils.data_utils.pad_array

Data Directory
^^^^^^^^^^^^^^^
The DeepChem data directory is where downloaded MoleculeNet datasets are stored.

.. autofunction:: deepchem.utils.data_utils.get_data_dir

URL Handling
^^^^^^^^^^^^

.. autofunction:: deepchem.utils.data_utils.download_url

File Handling
^^^^^^^^^^^^^

.. autofunction:: deepchem.utils.data_utils.untargz_file

.. autofunction:: deepchem.utils.data_utils.unzip_file

.. autofunction:: deepchem.utils.data_utils.load_data

.. autofunction:: deepchem.utils.data_utils.load_sdf_files

.. autofunction:: deepchem.utils.data_utils.load_csv_files

.. autofunction:: deepchem.utils.data_utils.load_json_files

.. autofunction:: deepchem.utils.data_utils.load_pickle_files

.. autofunction:: deepchem.utils.data_utils.load_from_disk

.. autofunction:: deepchem.utils.data_utils.save_to_disk

.. autofunction:: deepchem.utils.data_utils.load_dataset_from_disk

.. autofunction:: deepchem.utils.data_utils.save_dataset_to_disk

Molecular Utilities
-------------------

.. autoclass:: deepchem.utils.conformers.ConformerGenerator
  :members:

.. autoclass:: deepchem.utils.rdkit_utils.MoleculeLoadException
  :members:

.. autofunction:: deepchem.utils.rdkit_utils.get_xyz_from_mol

.. autofunction:: deepchem.utils.rdkit_utils.add_hydrogens_to_mol

.. autofunction:: deepchem.utils.rdkit_utils.compute_charges

.. autofunction:: deepchem.utils.rdkit_utils.load_molecule

.. autofunction:: deepchem.utils.rdkit_utils.write_molecule

Molecular Fragment Utilities
----------------------------

It's often convenient to manipulate subsets of a molecule. The :code:`MolecularFragment` class aids in such manipulations.

.. autoclass:: deepchem.utils.fragment_utils.MolecularFragment
  :members:

.. autoclass:: deepchem.utils.fragment_utils.AtomShim
  :members:

.. autofunction:: deepchem.utils.fragment_utils.strip_hydrogens

.. autofunction:: deepchem.utils.fragment_utils.merge_molecular_fragments

.. autofunction:: deepchem.utils.fragment_utils.get_contact_atom_indices

.. autofunction:: deepchem.utils.fragment_utils.reduce_molecular_complex_to_contacts

Coordinate Box Utilities
------------------------

.. autoclass:: deepchem.utils.coordinate_box_utils.CoordinateBox
  :members:

.. autofunction:: deepchem.utils.coordinate_box_utils.intersect_interval

.. autofunction:: deepchem.utils.coordinate_box_utils.union

.. autofunction:: deepchem.utils.coordinate_box_utils.merge_overlapping_boxes

.. autofunction:: deepchem.utils.coordinate_box_utils.get_face_boxes

Evaluation Utils
----------------

.. autoclass:: deepchem.utils.evaluate.Evaluator
  :members:

.. autoclass:: deepchem.utils.evaluate.GeneratorEvaluator
  :members:

.. autofunction:: deepchem.utils.evaluate.relative_difference


Genomic Utilities
-----------------

.. autofunction:: deepchem.utils.genomics_utils.seq_one_hot_encode

.. autofunction:: deepchem.utils.genomics_utils.encode_bio_sequence


Geometry Utilities
------------------

.. autofunction:: deepchem.utils.geometry_utils.unit_vector

.. autofunction:: deepchem.utils.geometry_utils.angle_between

.. autofunction:: deepchem.utils.geometry_utils.generate_random_unit_vector

.. autofunction:: deepchem.utils.geometry_utils.generate_random_rotation_matrix

.. autofunction:: deepchem.utils.geometry_utils.is_angle_within_cutoff

Hash Function Utilities
-----------------------

.. autofunction:: deepchem.utils.hash_utils.hash_ecfp

.. autofunction:: deepchem.utils.hash_utils.hash_ecfp_pair

.. autofunction:: deepchem.utils.hash_utils.vectorize

Voxel Utils
-----------

.. autofunction:: deepchem.utils.voxel_utils.convert_atom_to_voxel

.. autofunction:: deepchem.utils.voxel_utils.convert_atom_pair_to_voxel

.. autofunction:: deepchem.utils.voxel_utils.voxelize


Graph Convolution Utilities
---------------------------

.. autofunction:: deepchem.utils.molecule_feature_utils.one_hot_encode

.. autofunction:: deepchem.utils.molecule_feature_utils.get_atom_type_one_hot

.. autofunction:: deepchem.utils.molecule_feature_utils.construct_hydrogen_bonding_info

.. autofunction:: deepchem.utils.molecule_feature_utils.get_atom_hydrogen_bonding_one_hot

.. autofunction:: deepchem.utils.molecule_feature_utils.get_atom_is_in_aromatic_one_hot

.. autofunction:: deepchem.utils.molecule_feature_utils.get_atom_hybridization_one_hot

.. autofunction:: deepchem.utils.molecule_feature_utils.get_atom_total_num_Hs_one_hot

.. autofunction:: deepchem.utils.molecule_feature_utils.get_atom_chirality_one_hot

.. autofunction:: deepchem.utils.molecule_feature_utils.get_atom_formal_charge

.. autofunction:: deepchem.utils.molecule_feature_utils.get_atom_partial_charge

.. autofunction:: deepchem.utils.molecule_feature_utils.get_atom_total_degree_one_hot

.. autofunction:: deepchem.utils.molecule_feature_utils.get_bond_type_one_hot

.. autofunction:: deepchem.utils.molecule_feature_utils.get_bond_is_in_same_ring_one_hot

.. autofunction:: deepchem.utils.molecule_feature_utils.get_bond_is_conjugated_one_hot

.. autofunction:: deepchem.utils.molecule_feature_utils.get_bond_stereo_one_hot

.. autofunction:: deepchem.utils.molecule_feature_utils.get_bond_graph_distance_one_hot


Debug Utilities
---------------

Docking Utilities
-----------------

These utilities assist in file preparation and processing for molecular
docking.

.. autofunction:: deepchem.utils.docking_utils.write_vina_conf

.. autofunction:: deepchem.utils.docking_utils.write_gnina_conf

.. autofunction:: deepchem.utils.docking_utils.load_docked_ligands

.. autofunction:: deepchem.utils.docking_utils.prepare_inputs

.. autofunction:: deepchem.utils.docking_utils.read_gnina_log


Print Threshold
^^^^^^^^^^^^^^^

The printing threshold controls how many dataset elements are printed
when :code:`dc.data.Dataset` objects are converted to strings or
represnted in the IPython repl.

.. autofunction:: deepchem.utils.debug_utils.get_print_threshold

.. autofunction:: deepchem.utils.debug_utils.set_print_threshold

.. autofunction:: deepchem.utils.debug_utils.get_max_print_size

.. autofunction:: deepchem.utils.debug_utils.set_max_print_size

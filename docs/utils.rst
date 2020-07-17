Utilities
=========
DeepChem has a broad collection of utility functions. Many of these
maybe be of independent interest to users since they deal with some
tricky aspects of processing scientific datatypes.

Array Utilities
---------------

.. autofunction:: deepchem.utils.pad_array

Data Directory
--------------
The DeepChem data directory is where downloaded MoleculeNet datasets are stored.

.. autofunction:: deepchem.utils.get_data_dir

Print Threshold
---------------

The printing threshold controls how many dataset elements are printed
when :code:`dc.data.Dataset` objects are converted to strings or
represnted in the IPython repl.

.. autofunction:: deepchem.utils.get_print_threshold

.. autofunction:: deepchem.utils.set_print_threshold

.. autofunction:: deepchem.utils.get_max_print_size

.. autofunction:: deepchem.utils.set_max_print_size

URL Handling
------------

.. autofunction:: deepchem.utils.download_url

File Handling
-------------

.. autofunction:: deepchem.utils.untargz_file

.. autofunction:: deepchem.utils.unzip_file

.. autofunction:: deepchem.utils.save.save_to_disk

.. autofunction:: deepchem.utils.save.get_input_type

.. autofunction:: deepchem.utils.save.load_data

.. autofunction:: deepchem.utils.save.load_sharded_csv

.. autofunction:: deepchem.utils.save.load_sdf_files

.. autofunction:: deepchem.utils.save.load_csv_files

.. autofunction:: deepchem.utils.save.save_metadata

.. autofunction:: deepchem.utils.save.load_from_disk

.. autofunction:: deepchem.utils.save.load_pickle_from_disk

.. autofunction:: deepchem.utils.save.load_dataset_from_disk

.. autofunction:: deepchem.utils.save.save_dataset_to_disk

Molecular Utilities
-------------------

.. autoclass:: deepchem.utils.ScaffoldGenerator
  :members:

.. autoclass:: deepchem.utils.conformers.ConformerGenerator
  :members:

.. autoclass:: deepchem.utils.rdkit_util.MoleculeLoadException
  :members:

.. autofunction:: deepchem.utils.rdkit_util.get_xyz_from_mol

.. autofunction:: deepchem.utils.rdkit_util.add_hydrogens_to_mol

.. autofunction:: deepchem.utils.rdkit_util.compute_charges

.. autofunction:: deepchem.utils.rdkit_util.load_molecule

.. autofunction:: deepchem.utils.rdkit_util.write_molecule

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

.. autofunction:: deepchem.utils.evaluate.threshold_predictions

Genomic Utilities
-----------------

.. autofunction:: deepchem.utils.genomics.seq_one_hot_encode

.. autofunction:: deepchem.utils.genomics.encode_fasta_sequence

.. autofunction:: deepchem.utils.genomics.encode_bio_sequence


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

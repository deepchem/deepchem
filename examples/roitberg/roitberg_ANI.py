import numpy as np
import os

import deepchem as dc

import pyanitools as pya


def convert_species_to_atomic_nums(s):

  PERIODIC_TABLE = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "S": 16}

  res = []
  for k in s:
    res.append(PERIODIC_TABLE[k])
  return np.array(res, dtype=np.float32)


def load_roiterberg_ANI(relative=False):
  if "ROITBERG_ANI" not in os.environ:
    raise ValueError(
        "Please set environment variable ROITBERG_ANI to where the ani_dgb_s0x.h5 files are."
    )

  base_dir = os.environ["ROITBERG_ANI"]

  # Number of conformations in each file increases exponentially.
  # Start with a smaller dataset before continuing.
  hdf5files = [
      'ani_gdb_s01.h5',
      'ani_gdb_s02.h5',
      # 'ani_gdb_s03.h5',
      # 'ani_gdb_s04.h5',
      # 'ani_gdb_s05.h5',
      # 'ani_gdb_s06.h5',
      # 'ani_gdb_s07.h5',
      # 'ani_gdb_s08.h5'
  ]

  hdf5files = [os.path.join(base_dir, f) for f in hdf5files]

  groups = []

  atom_number_cases = (1, 6, 7, 8, 16)

  ANItransformer = dc.trans.ANITransformer(
      max_atoms=23, atom_cases=atom_number_cases)
  ANItransformer.transform_batch_size = 128

  def shard_generator():

    shard_size = 128 * 64

    row_idx = 0
    group_idx = 0

    X_cache = []
    y_cache = []
    w_cache = []
    ids_cache = []

    for hdf5file in hdf5files:
      adl = pya.anidataloader(hdf5file)
      for data in adl:

        # Extract the data
        P = data['path']
        R = data['coordinates']
        E = data['energies']
        S = data['species']
        smi = data['smiles']

        if len(S) > 23:
          print("skipping:", smi, "due to atom count.")
          continue

        # Print the data
        print("Processing: ", P)
        print("  Smiles:      ", "".join(smi))
        print("  Symbols:     ", S)
        print("  Coordinates: ", R.shape)
        print("  Energies:    ", E.shape)

        Z_padded = np.zeros((23,), dtype=np.float32)
        nonpadded = convert_species_to_atomic_nums(S)
        Z_padded[:nonpadded.shape[0]] = nonpadded

        if relative:
          offset = np.amin(E)
        else:
          offset = 0

        for k in range(len(E)):
          R_padded = np.zeros((23, 3), dtype=np.float32)
          R_padded[:R[k].shape[0], :R[k].shape[1]] = R[k]

          X = np.concatenate([np.expand_dims(Z_padded, 1), R_padded], axis=1)

          y = E[k] - offset  # offset is zero if we're not computing relative

          if len(X_cache) == shard_size:

            # flush when buffer is full
            tmp_dataset = dc.data.NumpyDataset(
                np.array(X_cache),
                np.array(y_cache), np.array(w_cache), np.array(ids_cache))

            tmp_dataset = ANItransformer.transform(tmp_dataset)

            yield tmp_dataset.X, tmp_dataset.y, tmp_dataset.w, tmp_dataset.ids

            X_cache = []
            y_cache = []
            w_cache = []
            ids_cache = []

          else:
            X_cache.append(X)
            y_cache.append(y)
            w_cache.append(1)
            ids_cache.append(row_idx)
            groups.append(group_idx)

          row_idx += 1

        group_idx += 1

    # flush once more at the end
    if len(X_cache) > 0:
      tmp_dataset = dc.data.NumpyDataset(
          np.array(X_cache),
          np.array(y_cache), np.array(w_cache), np.array(ids_cache))

      tmp_dataset = ANItransformer.transform(tmp_dataset)

      yield tmp_dataset.X, tmp_dataset.y, tmp_dataset.w, tmp_dataset.ids

  tasks = ["ani"]
  dataset = dc.data.DiskDataset.create_dataset(shard_generator(), tasks=tasks)
  ANItransformer.sess.close()

  print("Number of groups", np.amax(groups))
  print("dataset_shape", dataset.X.shape)
  splitter = dc.splits.RandomGroupSplitter(groups)
  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      dataset,
      frac_train=.5,
      frac_valid=.25,
      frac_test=.25,)

  transformers = [
      dc.trans.NormalizationTransformer(
          transform_y=True, dataset=train_dataset)
  ]

  for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    test_dataset = transformer.transform(test_dataset)

  return train_dataset, valid_dataset, test_dataset, transformers


if __name__ == "__main__":

  if "DEEPCHEM_DATA_DIR" in os.environ:
    data_dir = os.environ["DEEPCHEM_DATA_DIR"]
  else:
    data_dir = "/tmp"

  save_dir = os.path.join(data_dir, "roitberg")

  loaded, all_datasets, transformers = dc.utils.save.load_dataset_from_disk(
      save_dir)

  if loaded:
    train_dataset, valid_dataset, test_dataset = all_datasets
  else:
    train_dataset, valid_dataset, test_dataset, transformers = load_roiterberg_ANI(
    )
    print("Saving to disk...", save_dir)
    dc.utils.save.save_dataset_to_disk(save_dir, train_dataset, valid_dataset,
                                       test_dataset, transformers)

  max_atoms = 23
  batch_size = 256  # CHANGED FROM 16
  layer_structures = [128, 128, 64]
  atom_number_cases = [1, 6, 7, 8, 16]

  metric = [
      dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
      dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
  ]

  n_feat = dc.trans.ANITransformer(
      max_atoms=max_atoms, atom_cases=atom_number_cases).get_num_feats() - 1

  model = dc.models.ANIRegression(
      1,
      max_atoms,
      n_feat,
      layer_structures=layer_structures,
      atom_number_cases=atom_number_cases,
      batch_size=batch_size,
      learning_rate=0.001,
      use_queue=True,
      mode="regression")

  # Fit trained model
  for i in range(200):

    model.fit(train_dataset, nb_epoch=10, checkpoint_interval=100)

    print("Evaluating model")
    train_scores = model.evaluate(train_dataset, metric, transformers)
    valid_scores = model.evaluate(valid_dataset, metric, transformers)

    print("Train scores")
    print(train_scores)

    print("Validation scores")
    print(valid_scores)

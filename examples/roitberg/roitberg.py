import numpy as np
import os

import tensorflow as tf

import deepchem as dc
import pyanitools as pya
import app
import random

def convert_species_to_atomic_nums(s):
  PERIODIC_TABLE = {"H": 1, "C": 6, "N": 7, "O": 8}
  res = []
  for k in s:
    res.append(PERIODIC_TABLE[k])
  return np.array(res, dtype=np.float32)


# replace with your own scratch directory
data_dir = "/media/yutong/fast_datablob/datasets"
model_dir = "/media/yutong/fast_datablob/models"
# feat_train_dir = "/media/yutong/fast_datablob/feat"
# feat_valid_dir = "/media/yutong/fast_datablob/valid"

all_dir = os.path.join(data_dir, "all")
test_dir = os.path.join(data_dir, "test")
fold_dir = os.path.join(data_dir, "fold")
train_dir = os.path.join(fold_dir, "train")
temp_dir = os.path.join(fold_dir, "temp") # used for un-shuffled train data
valid_dir = os.path.join(fold_dir, "valid")


def load_roitberg_ANI(mode, batch_size):
  """
  Load the ANI dataset.

  Parameters
  ----------
  mode: str
    Accepted modes are "relative", "atomization", or "absolute". These settings are used
    to adjust the dynamic range of the model, with absolute having the greatest and relative
    having the lowest. Note that for atomization we approximate the single atom energy
    using a different level of theory


  Returns
  -------
  tuples
    Elements returned are 3-tuple (a,b,c) where and b are the train and test datasets, respectively,
    and c is an array of indices denoting the group of each 

  """
  if "ROITBERG_ANI" not in os.environ:
    raise ValueError(
        "Please set environment variable ROITBERG_ANI to where the ani_dgb_s0x.h5 files are."
    )

  print("FEATURIZATION MODE:", mode)

  base_dir = os.environ["ROITBERG_ANI"]

  # Number of conformations in each file increases exponentially.
  # Start with a smaller dataset before continuing. Use all of them
  # for production
  hdf5files = [
      'ani_gdb_s01.h5',
      'ani_gdb_s02.h5',
      'ani_gdb_s03.h5',
      'ani_gdb_s04.h5', # ~500k conformations
      # 'ani_gdb_s05.h5', # ~1.7e6 conformations
      # 'ani_gdb_s06.h5',
      # 'ani_gdb_s07.h5',
      # 'ani_gdb_s08.h5'
  ]

  hdf5files = [os.path.join(base_dir, f) for f in hdf5files]

  groups = []

  def shard_generator():

    shard_size = 4096 * batch_size

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

        if mode == "relative":
          offset = np.amin(E)
        elif mode == "atomization":

          # self-interaction energies taken from
          # https://github.com/isayev/ANI1_dataset README
          atomizationEnergies = {
              0: 0,
              1: -0.500607632585,
              6: -37.8302333826,
              7: -54.5680045287,
              8: -75.0362229210
          }

          offset = 0

          for z in nonpadded:
            offset += atomizationEnergies[z]
        elif mode == "absolute":
          offset = 0
        else:
          raise Exception("Unsupported mode: ", mode)

        for k in range(len(E)):
          R_padded = np.zeros((23, 3), dtype=np.float32)
          R_padded[:R[k].shape[0], :R[k].shape[1]] = R[k]

          X = np.concatenate([np.expand_dims(Z_padded, 1), R_padded], axis=1)

          y = E[k] - offset

          if len(X_cache) == shard_size:

            # (ytz): Note that this yields different shaped arrays
            yield np.array(X_cache), np.array(y_cache), np.array(
                w_cache), np.array(ids_cache)

            X_cache = []
            y_cache = []
            w_cache = []
            ids_cache = []

          X_cache.append(X)
          y_cache.append(np.array(y).reshape((1,)))
          w_cache.append(np.array(1).reshape((1,)))
          ids_cache.append(row_idx)
          row_idx += 1
          groups.append(group_idx)

        group_idx += 1

    # flush once more at the end
    if len(X_cache) > 0:

      yield np.array(X_cache), np.array(y_cache), np.array(w_cache), np.array(
          ids_cache)

  tasks = ["ani"]
  dataset = dc.data.DiskDataset.create_dataset(
      shard_generator(), tasks=tasks, data_dir=all_dir)

  print("Number of groups", np.amax(groups))
  splitter = dc.splits.RandomGroupSplitter(groups)

  train_dataset, test_dataset = splitter.train_test_split(
      dataset, train_dir=fold_dir, test_dir=test_dir, frac_train=.9)

  return train_dataset, test_dataset, groups



def broadcast(dataset, metadata):

  new_metadata = []

  for (_, _, _, ids) in dataset.itershards():
    for idx in ids:
      new_metadata.append(metadata[idx])

  return new_metadata

if __name__ == "__main__":

  max_atoms = 23
  batch_size = 384  # CHANGED FROM 192
  layer_structures = [128, 128, 64, 1]
  atom_number_cases = [1, 6, 7, 8]

  metric = [
      dc.metrics.Metric(dc.metrics.root_mean_squared_error, mode="regression"),
      dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
      dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
  ]

  # switch for datasets and models
  if os.path.exists(valid_dir) and \
     os.path.exists(test_dir) and \
     os.path.exists(train_dir):
    print("Restoring existing datasets...")
    train_dataset = dc.data.DiskDataset(data_dir=train_dir)
    valid_dataset = dc.data.DiskDataset(data_dir=valid_dir)
    test_dataset = dc.data.DiskDataset(data_dir=test_dir)

    print("Restoring featurizations...")
    for dd in [train_dataset, valid_dataset, test_dataset]:
      fp = os.path.join(dd.data_dir, "feat")
      if os.path.exists(fp):
        dd.feat_dataset = dc.data.DiskDataset(data_dir=fp)

  else:
    print("Generating datasets")

    train_valid_dataset, test_dataset, all_groups = load_roitberg_ANI(
        mode="atomization", batch_size=batch_size)

    splitter = dc.splits.RandomGroupSplitter(
        broadcast(train_valid_dataset, all_groups))

    print("Performing 1-fold split...")

    # (ytz): the 0.888888 is used s.t. 0.9*0.88888888 = 0.8, and we end up with a 80/10/10 split
    train_dataset, valid_dataset = splitter.train_test_split(
        train_valid_dataset, train_dir=temp_dir, test_dir=valid_dir, frac_train=0.8888888888)

    print("Shuffling training dataset...")
    train_dataset = train_dataset.complete_shuffle(data_dir=train_dir)

    print("Featurizing...")
    feat_batch_size = 384
    model = dc.models.ANIRegression(
        1,
        max_atoms,
        layer_structures=layer_structures,
        atom_number_cases=atom_number_cases,
        batch_size=feat_batch_size,
        learning_rate=None,
        use_queue=True,
        model_dir=model_dir,
        shift_exp=True,
        mode="regression")
    model.build()
    # model.save_numpy()

    for dd in [train_dataset, valid_dataset, test_dataset]:
      fp = os.path.join(dd.data_dir, "feat")
      dd.feat_dataset = model.featurize(dd, fp)

  # (ytz): I like big batches and I cannot lie
  train_batch_size = 1024

  # transformers = [
  #     dc.trans.NormalizationTransformer(
  #         transform_y=True, dataset=train_dataset)
  # ]


  # print("Transforming....")

  # for transformer in transformers:
  #   train_dataset = transformer.transform(train_dataset)
  #   valid_dataset = transformer.transform(valid_dataset)
  #   test_dataset = transformer.transform(test_dataset)

  print("Total training set shape: ", train_dataset.get_shape())

  # need to hit 0.003 RMSE hartrees for 2kcal/mol.

  best_val_score = 1e9

  for lr_idx, lr in enumerate([1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]):

    print ("\n\nTRAINING with learning rate:", lr, "\n\n")

    shift_exp = True

    if lr_idx == 0:
      model = dc.models.ANIRegression(
          1,
          max_atoms,
          layer_structures=layer_structures,
          atom_number_cases=atom_number_cases,
          batch_size=train_batch_size,
          learning_rate=lr,
          use_queue=True,
          model_dir=model_dir,
          shift_exp=shift_exp,
          mode="regression")
    else:
      model = dc.models.ANIRegression.load_numpy(
        model_dir=model_dir,
        override_kwargs = {
          "learning_rate": lr,
          "shift_exp": shift_exp,
          "batch_size":  train_batch_size})

    converged = False
    continuous_epochs = 0
    max_continuous_epochs = 100

    while not converged and continuous_epochs < max_continuous_epochs:
      model.fit(train_dataset, nb_epoch=1, checkpoint_interval=100)
      # print("Validation score:")
      val_score = model.evaluate(valid_dataset, metric)
      val_score = val_score['root_mean_squared_error']
      print("This epoch's validation score:", val_score)
      if val_score < best_val_score:
        print("--------- Better validation score found:", val_score, "---------")
        best_val_score = val_score
        model.save_numpy()
        continuous_epochs = 0
      else:
        continuous_epochs += 1

  print("--train--")
  model.evaluate(train_dataset, metric)
  print("--valid--")
  model.evaluate(valid_dataset, metric)
  print("--test--")
  model.evaluate(test_dataset, metric)


  coords = np.array([
      [0.3, 0.4, 0.5],
      [0.8, 0.2, 0.3],
      [0.1, 0.3, 0.8],
  ])

  atomic_nums = np.array([1, 8, 1])

  print("Prediction of a single test set structure:")
  print(model.pred_one(coords, atomic_nums))

  print("Gradient of a single test set structure:")
  print(model.grad_one(coords, atomic_nums))

  # print("Minimization of a single test set structure:")
  # print(model.minimize_structure(coords, atomic_nums))

  app.webapp.model = model
  app.webapp.run(host='0.0.0.0', debug=False)

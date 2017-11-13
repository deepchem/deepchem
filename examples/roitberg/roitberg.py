#!/usr/bin/env python

import argparse
import errno
import numpy as np
import os
import shutil
import sys

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

def path(*dir_segments):
  return os.path.abspath(os.path.expanduser(os.path.join(*dir_segments)))

def path_not_empty(d):
  d = path(d)
  return os.path.exists(d) and len(os.listdir(d)) > 1

def find_or_create_path(*dir_segments):
  """
  Pretty much what it sounds like: finds a directory path or creates one if it
  doesn't exist.

  Also adds in the functionality of os.path.join, for extra helpfulness: the
  dir_segments will be joined together, then checked for existence.
  """
  d = path(*dir_segments)

  if not os.path.exists(d):
    os.makedirs(d, exist_ok=True)

  return d

def setup_work_dirs(args):
  """
  Sets up the work directory structure beneath the directory in +args.work_dir+.

  If args.work_dir is not a valid directory, errors out with a warning to the user,
  on the theory that the program shouldn't be chucking lots of data in places the
  user isn't explicitly aware of.

  Below args.work_dir, creates new subdirectories with impunity. The various work
  directories will be set in +args+ for later reference.

  Parameters:
    args - the result of calling ArgumentParser.parse_args()
  """
  args.work_dir = path(args.work_dir)

  if not os.path.exists(args.work_dir):
    # we don't create the work dir if it doesn't already exist:
    # don't want to silently dump big files on the user's FS without
    # their knowledge.
    sys.stderr.write("Work directory '%s' does not exist. "
                     "Create it, or specify a new directory with the -w option.\n" % args.work_dir)
    sys.exit(-1)
  elif args.clean_work_dir:
    # python doesn't have an easy way to remove all files in a directory
    # this will *probably* work, but might fail on weird setups.
    shutil.rmtree(args.work_dir)
    os.makedirs(args.work_dir, exist_ok=True)

  args.data_dir = find_or_create_path(args.work_dir, "datasets")
  args.model_dir = find_or_create_path(args.work_dir, "models")

  args.all_dir = find_or_create_path(args.data_dir, "all")
  args.test_dir = find_or_create_path(args.data_dir, "test")
  args.fold_dir = find_or_create_path(args.data_dir, "fold")

  args.train_dir = find_or_create_path(args.fold_dir, "train")
  args.temp_dir = find_or_create_path(args.fold_dir, "temp")
  args.valid_dir = find_or_create_path(args.fold_dir, "valid")

  return args


def find_training_data(base_dir, max_gdb_level):
  """
  Generates filenames for GDB files and checks for their existence.
  Fails out with user error if any file is not found.

  Returns:
    list of validated file path strings.
  """
  base_dir = os.path.abspath(os.path.expanduser(base_dir))
  files = [os.path.join(base_dir, "ani_gdb_s%02d.h5" % i) for i in range(max_gdb_level, 0, -1)]

  for f in files:
    if not os.path.exists(f):
      sys.stderr.write("Training data file '%s' not found." % f)
      sys.exit(-1)

  return files

def load_roitberg_ANI(args, mode="atomization"):
  """
  Load the ANI dataset.

  Parameters
  ----------
  args:
    Result of calling ArgumentParser.parse_args()

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

  hdf5files = find_training_data(args.training_data_dir, args.gdb_level)
  groups = []

  def shard_generator():

    shard_size = 4096 * args.batch_size

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
      yield np.array(X_cache), np.array(y_cache), np.array(w_cache), np.array(ids_cache)

  tasks = ["ani"]
  dataset = dc.data.DiskDataset.create_dataset(
      shard_generator(), tasks=tasks, data_dir=args.all_dir)

  print("Number of groups", np.amax(groups))
  splitter = dc.splits.RandomGroupSplitter(groups)

  train_dataset, test_dataset = splitter.train_test_split(
    dataset, train_dir=args.fold_dir, test_dir=args.test_dir, frac_train=.9)

  return train_dataset, test_dataset, groups



def broadcast(dataset, metadata):

  new_metadata = []

  for (_, _, _, ids) in dataset.itershards():
    for idx in ids:
      new_metadata.append(metadata[idx])

  return new_metadata

def parse_args():
  """
  Parses the command line arguments.
  """
  parser = argparse.ArgumentParser(description="Run ANI1 neural net training.",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-b', '--batch-size', type=int, default=1024,
                      help="training batch size")
  parser.add_argument('--featurization-batch-size', type=int, default=384,
                      help="featurization batch size")
  parser.add_argument('-e', '--max-search-epochs', type=int, default=100,
                      help="The maximum number of epochs to run w/o validation score improvement.")
  parser.add_argument('-v', '--validation-interval', type=int, default=10,
                      help="Validate the model every validation-interval epochs.")
  parser.add_argument('-w', '--work-dir', default='~/roitberg-scratch',
                      help="location where work data is dumped")
  parser.add_argument('-t', '--training-data-dir', default='~/roitberg-ani',
                      help="directory containing training/gdb data")
  parser.add_argument('--gdb-level', type=int, default=5,
                      help="Max GDB level to train. NOTE: num conformations " \
                      "in each file increases exponentially. Start with a smaller dataset. " \
                      "Use max value (8) for production.")
  parser.add_argument('--max-atoms', type=int, default=23,
                      help="max molecule size")
  parser.add_argument('--clean-work-dir', action='store_true',
                      help="Clean the work dir before training (i.e. do a clean run)")
  parser.add_argument('--restore-model', action='store_true',
                      help="Try to reload a previously saved model from the work dir.")

  return parser.parse_args()

#
# Program main
#
if __name__ == "__main__":
  args = parse_args()
  setup_work_dirs(args)

  max_atoms = args.max_atoms
  batch_size = args.batch_size
  layer_structures = [128, 128, 64, 1]
  atom_number_cases = [1, 6, 7, 8]

  metric = [
      dc.metrics.Metric(dc.metrics.root_mean_squared_error, mode="regression"),
      dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
      dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
  ]

  # switch for datasets and models
  if path_not_empty(args.valid_dir) and \
     path_not_empty(args.test_dir) and \
     path_not_empty(args.train_dir):

    print("Restoring existing datasets...")
    train_dataset = dc.data.DiskDataset(data_dir=args.train_dir)
    valid_dataset = dc.data.DiskDataset(data_dir=args.valid_dir)
    test_dataset = dc.data.DiskDataset(data_dir=args.test_dir)

    print("Restoring featurizations...")
    for dd in [train_dataset, valid_dataset, test_dataset]:
      fp = path(dd.data_dir, "feat")
      if path_not_empty(fp):
        dd.feat_dataset = dc.data.DiskDataset(data_dir=fp)

  else:
    print("Generating datasets...")

    train_valid_dataset, test_dataset, all_groups = load_roitberg_ANI(args, "atomization")

    splitter = dc.splits.RandomGroupSplitter(
        broadcast(train_valid_dataset, all_groups))

    print("Performing 1-fold split...")

    # (ytz): the 0.888888 is used s.t. 0.9*0.88888888 = 0.8, and we end up with a 80/10/10 split
    train_dataset, valid_dataset = splitter.train_test_split(
      train_valid_dataset, train_dir=args.temp_dir,
      test_dir=args.valid_dir, frac_train=0.8888888888)

    print("Shuffling training dataset...")
    train_dataset = train_dataset.complete_shuffle(data_dir=args.train_dir)

    print("Featurizing...")
    model = dc.models.ANIRegression(
        1,
        max_atoms,
        layer_structures=layer_structures,
        atom_number_cases=atom_number_cases,
        batch_size=args.featurization_batch_size,
        learning_rate=None,
        use_queue=True,
        model_dir=args.model_dir,
        shift_exp=True,
        mode="regression")
    model.build()

    for dd in [train_dataset, valid_dataset, test_dataset]:
      dd.feat_dataset = model.featurize(dd, path(dd.data_dir, "feat"))

  print("Total training set shape: ", train_dataset.get_shape())

  # need to hit 0.003 RMSE hartrees for 2kcal/mol.
  best_val_score = 1e9

  for lr_idx, lr in enumerate([1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]):
    print ("\n\nTRAINING with learning rate:", lr, "\n\n")

    if lr_idx == 0:
      model = dc.models.ANIRegression(
          1,
          max_atoms,
          layer_structures=layer_structures,
          atom_number_cases=atom_number_cases,
          batch_size=args.batch_size,
          learning_rate=lr,
          use_queue=True,
          model_dir=args.model_dir,
          shift_exp=True,
          mode="regression")
    else:
      model = dc.models.ANIRegression.load_numpy(
        model_dir=args.model_dir,
        override_kwargs = {
          "learning_rate": lr,
          "shift_exp": True,
          "batch_size": args.batch_size
        })

    epoch_count = 0

    while epoch_count < args.max_search_epochs:
      model.fit(train_dataset, nb_epoch=1, checkpoint_interval=100)
      val_score = model.evaluate(valid_dataset, metric)
      val_score = val_score['root_mean_squared_error']

      print("This epoch's validation score:", val_score)

      if val_score < best_val_score:
        print("--------- Better validation score found:", val_score, "---------")
        best_val_score = val_score
        model.save_numpy()
        epoch_count = 0
      else:
        epoch_count += 1

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

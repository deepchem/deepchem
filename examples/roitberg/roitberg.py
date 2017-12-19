#!/usr/bin/env python
import pickle
import cProfile
pr = cProfile.Profile()
pr.disable()
import pstats
from io import StringIO

import argparse
import math
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

HARTREE_TO_KCAL_PER_MOL = 627.509

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
  args.select_dir = find_or_create_path(args.data_dir, "select")
  args.test_dir = find_or_create_path(args.data_dir, "test")
  args.fold_dir = find_or_create_path(args.data_dir, "fold")
  args.gdb10_dir = find_or_create_path(args.data_dir, "gdb10")

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
  files = [os.path.join(base_dir, "ani_gdb_s%02d.h5" % i) for i in range(1, max_gdb_level+1)]

  for f in files:
    if not os.path.exists(f):
      sys.stderr.write("Training data file '%s' not found." % f)
      sys.exit(-1)

  return files


def find_gdb10_test_data(base_dir):
  base_dir = os.path.abspath(os.path.expanduser(base_dir))
  file = os.path.join(base_dir, "ani1_gdb10_ts.h5")
  return [file]


def load_hdf5_files(
  hdf5files,
  batch_size,
  data_dir,
  mode,
  max_atoms,
  energy_cutoff=100.0/HARTREE_TO_KCAL_PER_MOL,
  selection_size=None):
  """
  Load the ANI dataset.

  Parameters
  ----------
  hdf5files: list of str
    List of paths to hdf5 files that will be used to generate the dataset. The data should be
    in the format used by the ANI-1 dataset.

  batch_size: int
    Used to determined the shard_size, where shard_size is batch_size * 4096

  data_dir: str
    Directory in which we save the resulting data

  mode: str
    Accepted modes are "relative", "atomization", or "absolute". These settings are used
    to adjust the dynamic range of the model, with absolute having the greatest and relative
    having the lowest. Note that for atomization we approximate the single atom energy
    using a different level of theory

  max_atoms: int
    Total number of atoms we allow for.

  energy_cutoff: int or None
    A cutoff to use for pruning high energy conformations from a dataset. Units are in
    hartrees. Default is set to 100 kcal/mol or ~0.16 hartrees.

  selection_size: int or None
    Subsample of conformations that we want to choose from gdb-8

  Returns
  -------
  Dataset, list of int
    Returns a Dataset object and a list of integers corresponding to the groupings of the
    respective atoms.

  """
  groups = []

  def shard_generator(sel_size):

    shard_size = 4096 * batch_size

    X_cache = []
    y_cache = []
    w_cache = []
    ids_cache = []

    # loop once to compute the size of the dataset
    total_size = 0


    print("Counting total size and gathering statistics...")

    all_ys = []
    keep_rows = []

    skip_count = 0

    row_idx = 0
    for hdf5file in hdf5files:
      adl = pya.anidataloader(hdf5file)
      for data in adl:
        S = data['species']
        E = data['energies']
        if len(S) > max_atoms:
          # skip due to too many atoms.
          row_idx += len(E) # skip all the conformations for this smiles
          continue

        minimum = np.amin(E) # but this isn't the atomization

        if mode == "relative":
          offset = minimum
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
          nonpadded = convert_species_to_atomic_nums(S)
          for z in nonpadded:
            offset += atomizationEnergies[z]
        elif mode == "absolute":
          offset = 0
        else:
          raise Exception("Unsupported mode: ", mode)

        for k in range(len(E)):
          if energy_cutoff is not None and E[k] - minimum > energy_cutoff:
            # skip due to high energy
            # print("skipping",  y - minimum_atomization, energy_cutoff, E[k], minimum)
            skip_count += 1
            row_idx += 1
            continue

          y = E[k] - offset
          all_ys.append(y)
          keep_rows.append(row_idx)

          row_idx += 1

    total_size = len(keep_rows)

    if sel_size is None:
      sel_size = total_size

    assert isinstance(sel_size, int)

    print(total_size, "valid elements out of", row_idx, "keeping", sel_size)

    keep_flags = np.zeros(row_idx, dtype=np.bool) # boolean flag of total size
    keep_ys = np.zeros(row_idx, dtype=np.float32) # energies
    # to total number of molecules

    # note that we still need a true random shuffle after (outside since this still
    # has an ordering dependence). 
    keep_idxs = np.random.permutation(total_size)[:sel_size]

    assert len(keep_idxs) == sel_size

    all_ys = np.array(all_ys)
    keep_rows = np.array(keep_rows)

    all_ys = all_ys[keep_idxs]
    keep_rows = keep_rows[keep_idxs]

    keep_ys[keep_rows] = all_ys
    keep_flags[keep_rows] = 1

    print("Starting iteration")

    old_size = row_idx

    row_idx = 0
    group_idx = 0
    for hdf5file in hdf5files:
      adl = pya.anidataloader(hdf5file)
      for data in adl:

        # Extract the data
        P = data['path']
        R = data['coordinates']
        E = data['energies']
        S = data['species']
        smi = data['smiles']

        if len(S) > max_atoms:
          print("skipping ", smi, "due to atom count: ", len(S))
          group_idx += 1
          row_idx += len(E)
          continue

        # Print the data
        print("Processing: ", P)
        # print("  Smiles:      ", "".join(smi))
        # print("  Symbols:     ", S)
        # print("  Coordinates: ", R.shape)
        # print("  Energies:    ", E.shape)

        Z_padded = np.zeros((max_atoms,), dtype=np.float32)
        nonpadded = convert_species_to_atomic_nums(S)
        Z_padded[:nonpadded.shape[0]] = nonpadded

        minimum = np.amin(E)

        for k in range(len(E)):
          R_padded = np.zeros((max_atoms, 3), dtype=np.float32)
          R_padded[:R[k].shape[0], :R[k].shape[1]] = R[k]

          X = np.concatenate([np.expand_dims(Z_padded, 1), R_padded], axis=1)

          if len(X_cache) == shard_size:

            # (ytz): Note that this yields different shaped arrays
            yield np.array(X_cache), np.array(y_cache), np.array(
                w_cache), np.array(ids_cache)

            X_cache = []
            y_cache = []
            w_cache = []
            ids_cache = []

          if keep_flags[row_idx]:
            X_cache.append(X)
            y = keep_ys[row_idx]
            y_cache.append(np.array(y).reshape((1,)))
            w_cache.append(np.array(1).reshape((1,)))
            ids_cache.append(row_idx)
            groups.append(group_idx)

          row_idx += 1

        group_idx += 1

    assert row_idx == old_size

    # flush once more at the end
    if len(X_cache) > 0:
      yield np.array(X_cache), np.array(y_cache), np.array(w_cache), np.array(ids_cache)

  tasks = ["ani"]
  dataset = dc.data.DiskDataset.create_dataset(
      shard_generator(selection_size), tasks=tasks, data_dir=data_dir)

  if selection_size:
    print("DATASET LENGTH", len(dataset))
    assert len(dataset) == selection_size

  print("Skipped")
  return dataset, groups

def load_roitberg_ANI(args, mode):
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

  all_train = find_training_data(args.training_data_dir, args.gdb_level)
  all_dataset, groups = load_hdf5_files(
    all_train,
    batch_size=args.batch_size,
    data_dir=args.all_dir,
    mode=mode,
    max_atoms=args.max_atoms,
    selection_size=int(1e6))

  # split based on chemotype for true test set generalizibility
  # splitter = dc.splits.RandomGroupSplitter(groups)
  splitter = dc.splits.RandomSplitter(all_dataset)

  train_dataset, test_dataset = splitter.train_test_split(
    all_dataset, train_dir=args.fold_dir, test_dir=args.test_dir, frac_train=.9)

  gdb10_test = find_gdb10_test_data(args.training_data_dir)
  gdb10_dataset, gdb10_groups = load_hdf5_files(
    gdb10_test,
    args.batch_size,
    args.gdb10_dir,
    mode,
    args.max_atoms)

  return train_dataset, test_dataset, groups, gdb10_dataset, gdb10_groups

def broadcast(dataset, metadata):

  new_metadata = []

  for (_, _, _, ids) in dataset.itershards():
    for idx in ids:
      new_metadata.append(metadata[idx])

  return new_metadata

def die(msg, retval=-1):
  """Terminate the program with a warning message."""
  sys.stderr.write("Execution failed with error:\n\n%s" % msg)
  sys.exit(retval)

def parse_args():
  """
  Parses the command line arguments.

  Returns:

  An argparse.Namespace object, as returned by ArgumentParser.parse_args().
  This object will have at least the following fields:

    * batch_size int
      The batch size for training.
    * featurization_batch_size int
      The batch size for featurization
    * max_search_epochs int
      The the maximum number of epochs to search for a lower validation score.
    * initial_learning_rate float
      The starting learning rate for a new model.
    * learning_rate_factor float
      The factor by which to reduce the learning rate as training proceeds.
    * work_dir string
      The path of the working directory (i.e. where we write stuff).
    * training_data_dir string
      The path of the directory holding the input data for training.
    * gdb_level int
      The maximum GDB level to use for training.
    * max_atoms int
      The maximum molecule size.
    * clean_work_dir bool
      If true, indicates that the working directory should be wiped before
      training.
    * reload_model bool
      If true, indicates that a model should be reloaded instead of training
      a new one from scratch.
  """
  parser = argparse.ArgumentParser(description="Run ANI1 neural net training.",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('-b', '--batch-size', type=int, default=1024,
                      help="training batch size")
  parser.add_argument('--featurization-batch-size', type=int, default=256,
                      help="featurization batch size")
  parser.add_argument('-e', '--max-search-epochs', type=int, default=100,
                      help="The maximum number of epochs to run w/o validation score improvement.")
  parser.add_argument('-ilr', '--initial-learning-rate', type=float, default=0.001,
                      help="The initial learning rate.")
  parser.add_argument('-lrf', '--learning-rate-factor', type=float, default=10.0,
                      help="The factor by which to reduce the learning rate every --max-search-epochs")

  parser.add_argument('-w', '--work-dir', default='~/roitberg-scratch',
                      help="location where work data is dumped")
  parser.add_argument('-t', '--training-data-dir', default='~/roitberg-ani',
                      help="directory containing training/gdb data")
  parser.add_argument('--gdb-level', type=int, default=5,
                      help="Max GDB level to train. NOTE: num conformations " \
                      "in each file increases exponentially. Start with a smaller dataset. " \
                      "Use max value (8) for production.")
  parser.add_argument('--max-atoms', type=int, default=24,
                      help="max molecule size")
  parser.add_argument('--clean-work-dir', action='store_true',
                      help="Clean the work dir before training (i.e. do a clean run)")
  parser.add_argument('--reload-model', action='store_true',
                      help="Reloads an existing model from the work dir, instead of training new.")

  args = parser.parse_args()

  if args.batch_size % args.featurization_batch_size != 0:
    die("--batch-size must be evenly divisible by --featurization-batch-size!")

  if args.max_atoms != 24:
    # just for now, I hope...
    die("You can use any --max-atoms you like, as long as it's 24.")

  if args.initial_learning_rate < 0:
    die("The --initial-learning-rate must be a positive number.")

  if args.learning_rate_factor <= 1.0:
    die("The --learning-rate-factor must be greater than 1.")

  return args

#
# Program main
#
if __name__ == "__main__":
  args = parse_args()
  setup_work_dirs(args)

  max_atoms = args.max_atoms
  batch_size = args.batch_size
  train_batch_size = args.featurization_batch_size * 3
  layer_structures = [256, 128, 64, 1]
  atom_number_cases = [1, 6, 7, 8]

  metric = [
      dc.metrics.Metric(dc.metrics.root_mean_squared_error, mode="regression"),
      dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
      dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
  ]

  # switch for datasets and models

  if path_not_empty(args.train_dir) and \
     path_not_empty(args.valid_dir) and \
     path_not_empty(args.test_dir) and \
     path_not_empty(args.gdb10_dir):

    print("Restoring existing datasets...")
    train_dataset = dc.data.DiskDataset(data_dir=args.train_dir)
    valid_dataset = dc.data.DiskDataset(data_dir=args.valid_dir)
    test_dataset = dc.data.DiskDataset(data_dir=args.test_dir)
    gdb10_dataset = dc.data.DiskDataset(data_dir=args.gdb10_dir)

    print("Restoring featurizations...")
    for dd in [train_dataset, valid_dataset, test_dataset, gdb10_dataset]:
      fp = path(dd.data_dir, "feat")
      if path_not_empty(fp):
        dd.feat_dataset = dc.data.DiskDataset(data_dir=fp)

  else:
    print("Generating train_valid/test datasets...")

    train_valid_dataset, test_dataset, all_groups, gdb10_dataset, gdb10_groups = load_roitberg_ANI(args, "atomization")

    # splitter = dc.splits.RandomGroupSplitter(
        # broadcast(train_valid_dataset, all_groups))

    # train/valid split uses completely random split because we don't want to overfit
    # to a specific set of chemotypes.
    # splitter = dc.splits.RandomSplitter(
        # broadcast(train_valid_dataset, all_groups))
    splitter = dc.splits.RandomSplitter(train_valid_dataset)

    print("Performing 1-fold split...")

    # (ytz): the 0.888888 is used s.t. 0.9*0.88888888 = 0.8, and we end up with a 80/10/10 split
    temp_train_dataset, valid_dataset = splitter.train_test_split(
      train_valid_dataset, train_dir=args.temp_dir,
      test_dir=args.valid_dir, frac_train=0.8888888888)

    print("Shuffling training dataset...")
    train_dataset = temp_train_dataset.complete_shuffle(data_dir=args.train_dir)

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

    for dd in [train_dataset, valid_dataset, test_dataset, gdb10_dataset]:
      print("Featurizing into:", dd.data_dir)
      dd.feat_dataset = model.featurize(dd, path(dd.data_dir, "feat"))

  print("Train, Valid, Test, GDB10 sizes:", len(train_dataset), len(valid_dataset), len(test_dataset), len(gdb10_dataset))
 
  # need to hit 0.003 RMSE hartrees for 2kcal/mol.
  best_val_score = 1e9
  lr_idx = 0
  lr = args.initial_learning_rate

  print("Training from initial learning rate %f, with step factor %f" % (lr, args.learning_rate_factor))

  max_batches = math.ceil(len(train_dataset) / args.batch_size)

  while lr >= 1e-9:
    print ("\n\nTRAINING with learning rate:", lr, "\n\n")

    if args.reload_model or lr_idx != 0:
      model = dc.models.ANIRegression.load_numpy(
        model_dir=args.model_dir,
        override_kwargs = {
          "learning_rate": lr,
          "shift_exp": True,
          "batch_size": args.batch_size
        })
    else:
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

    # ############# DEBUG


    # gdb10_test = find_gdb10_test_data(args.training_data_dir)
    # gdb10_tmp, gdb10_groups = load_hdf5_files(
    #   gdb10_test,
    #   args.batch_size,
    #   args.gdb10_dir,
    #   "atomization",
    #   args.max_atoms)

    # print("GDB10_GROUPS", gdb10_groups)

    # val_score = model.evaluate(gdb10_dataset, metric)
    # print(val_score)

    # assert 0
    #############

    if lr_idx == 0:
      val_score = model.evaluate(valid_dataset, metric)
      best_val_score = val_score['root_mean_squared_error']

    epoch_count = 0

    print("....")

    while epoch_count < args.max_search_epochs:
      # pr.enable()
      print("fitting....", len(train_dataset))
      model.fit(
        train_dataset,
        nb_epoch=1,
        checkpoint_interval=0,
        max_batches=max_batches
      )
      print("fitting done...")

      print("--val--")
      val_score = model.evaluate(valid_dataset, metric)
      val_score = val_score['root_mean_squared_error']
      print("val score in kcal/mol:", val_score*HARTREE_TO_KCAL_PER_MOL)

      print("--test--")
      test_score = model.evaluate(test_dataset, metric)
      test_score = test_score['root_mean_squared_error']
      print("test score in kcal/mol:", test_score*HARTREE_TO_KCAL_PER_MOL)

      print("--gdb10--")
      gdb10_score = model.evaluate(gdb10_dataset, metric)
      gdb10_score = gdb10_score['root_mean_squared_error']
      print("test score in kcal/mol:", gdb10_score*HARTREE_TO_KCAL_PER_MOL)      

      print("This epoch's validation score:", val_score)

      if val_score < best_val_score:
        print("--------- Better validation score found:", val_score, "---------")
        best_val_score = val_score
        model.save_numpy()
        epoch_count = 0
      else:
        epoch_count += 1

    # pr.disable()

    # # if epoch_count == 0:
    # s = StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())

    lr /= args.learning_rate_factor
    lr_idx += 1

  print("--train--")
  model.evaluate(train_dataset, metric)
  print("--valid--")
  model.evaluate(valid_dataset, metric)
  print("--test--")
  model.evaluate(test_dataset, metric)
  print("--gdb10--")
  model.evaluate(gdb10_dataset, metric)


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

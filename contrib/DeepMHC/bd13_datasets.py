"""BD2013 dataset loader to be used with DeepMHC."""

from __future__ import division
from __future__ import print_function

__author__ = "Vignesh Ram Somnath"
__license__ = "MIT"

import numpy as np
import os
import logging

import deepchem as dc

DATASET_URL = "http://tools.iedb.org/static/main/binding_data_2013.zip"
FILE_NAME = "bdata.20130222.mhci.txt"

TEST_FILES = [
    "2016-12-09", "2016-05-03", "2016-02-19", "2015-08-07", "2015-07-31",
    "2015-07-17", "2015-06-26", "2015-06-19", "2015-05-15", "2015-02-06",
    "2015-01-16", "2014-10-31", "2014-06-20", "2014-05-23", "2014-03-28",
    "2014-03-21"
]

TEST_URLS = [
    "http://tools.iedb.org/auto_bench/mhci/weekly/accumulated/" + str(date) +
    "/predictions" for date in TEST_FILES
]

logger = logging.getLogger(__name__)

aa_charset = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P",
    "S", "T", "W", "Y", "V"
]


def to_one_hot_array(sequence):
  """Converts the sequence to one-hot-array."""
  one_hot_array = list()
  for letter in sequence:
    one_hot_array.append([letter == i for i in aa_charset])
  return np.asarray(one_hot_array, dtype=np.int32)


def featurize(sequences, pad_length=13):
  """One-hot encoding for sequences with padding."""
  features = list()
  for sequence in sequences:
    one_hot_seq = to_one_hot_array(sequence)
    num_to_pad = pad_length - len(sequence)
    if num_to_pad % 2 == 0:
      one_hot_seq = np.pad(
          one_hot_seq, [(int(num_to_pad / 2), int(num_to_pad / 2)), (0, 0)],
          mode='constant')
    else:
      one_hot_seq = np.pad(
          one_hot_seq, [(int((num_to_pad + 1) / 2), int((num_to_pad - 1) / 2)),
                        (0, 0)],
          mode='constant')
    features.append(one_hot_seq)
  features = np.asarray(features)
  return features


def load_bd2013_human(mhc_allele="HLA-A*02:01",
                      seq_len=9,
                      pad_len=13,
                      test_measure_type="ic50",
                      reload=True):
  """Loads the human specific data from the bd2013 dataset."""
  bd13_tasks = ["-log(IC50)"]

  data_dir = dc.utils.get_data_dir()
  save_dir = os.path.join(data_dir, "bd13", mhc_allele, str(seq_len))
  train_dir = os.path.join(save_dir, "train_dir")
  test_dir = os.path.join(save_dir, "test_dir")

  # TODO (VIGS25): Account for the reload option

  # Downloading train files
  train_file = os.path.join(data_dir, "binding_data_2013.zip")
  if not os.path.exists(train_file):
    logger.info("Downloading Binding data...")
    dc.utils.download_url(url=DATASET_URL, dest_dir=data_dir)
  if os.path.exists(train_dir):
    logger.info("Directory for training data already exists")
  else:
    logger.info("Unzipping full dataset...")
    dc.utils.unzip_file(file=train_file, dest_dir=data_dir)

  # Parsing training data
  train_labels = list()
  train_sequences = list()
  with open(os.path.join(data_dir, FILE_NAME), "r") as f:
    for line in f.readlines():
      elements = line.strip().split("\t")
      # Pick only sequences from humans, belong to specific MHC allele and having given seq_len
      if elements[0] == "human" and elements[1] == mhc_allele and int(
          elements[2]) == seq_len:
        train_sequences.append(elements[3])
        train_labels.append(float(elements[-1]))

  # Test Files loading
  test_labels = list()
  test_sequences = list()
  test_check_file = os.path.join(data_dir, TEST_FILES[0] + '_predictions.tsv')
  if not os.path.exists(test_check_file):
    for index, filename in enumerate(TEST_FILES):
      test_url = TEST_URLS[index]
      test_filename = filename + '_predictions.tsv'
      dc.utils.download_url(url=test_url, dest_dir=data_dir, name=test_filename)

  for filename in TEST_FILES:
    test_filename = os.path.join(data_dir, filename + '_predictions.tsv')
    with open(test_filename, 'r') as f:
      for line in f.readlines():
        elements = line.strip().split("\t")
        if len(elements) == 1:
          continue
        if elements[2] == mhc_allele and int(
            elements[3]) == seq_len and elements[4] == test_measure_type:
          test_sequences.append(elements[5])
          test_labels.append(float(elements[6]))

  # One Hot Featurization
  logger.info("Featurizing training data...")
  train_features = featurize(train_sequences, pad_length=pad_len)
  train_labels = np.array(train_labels).astype(np.float32)
  train_labels = np.expand_dims(train_labels, axis=1)

  logger.info("Featurizing test data...")
  test_features = featurize(test_sequences, pad_length=pad_len)
  test_labels = np.array(test_labels).astype(np.float32)
  test_labels = np.expand_dims(test_labels, axis=1)

  train_dataset = dc.data.DiskDataset.from_numpy(train_features, train_labels)
  test_dataset = dc.data.DiskDataset.from_numpy(test_features, test_labels)

  train_dataset.move(new_data_dir=train_dir)
  test_dataset.move(new_data_dir=test_dir)

  logger.info("Featurization complete.")

  transformers = []
  for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    test_dataset = transformer.transform(test_dataset)

  return bd13_tasks, (train_dataset, None, test_dataset), transformers

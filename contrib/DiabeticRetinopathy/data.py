"""
Diabetic Retinopathy Images loader.
"""
from __future__ import division
from __future__ import unicode_literals

import os
import logging
import deepchem
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_images_DR(split='random', seed=None):
  """ Loader for DR images """
  data_dir = deepchem.utils.get_data_dir()
  images_path = os.path.join(data_dir, 'DR', 'train')
  label_path = os.path.join(data_dir, 'DR', 'trainLabels.csv')
  if not os.path.exists(images_path) or not os.path.exists(label_path):
    logger.warn("Cannot locate data, \n\
        all images(.png) should be stored in the folder: $DEEPCHEM_DATA_DIR/DR/train/,\n\
        corresponding label file should be stored as $DEEPCHEM_DATA_DIR/DR/trainLabels.csv.\n\
        Please refer to https://www.kaggle.com/c/diabetic-retinopathy-detection for data access"
               )

  image_names = os.listdir(images_path)
  raw_images = []
  for im in image_names:
    if im.endswith('.jpeg') and not im.startswith(
        'cut_') and not 'cut_' + im in image_names:
      raw_images.append(im)
  if len(raw_images) > 0:
    cut_raw_images(raw_images, images_path)

  image_names = [
      p for p in os.listdir(images_path)
      if p.startswith('cut_') and p.endswith('.png')
  ]
  all_labels = dict(zip(*np.transpose(np.array(pd.read_csv(label_path)))))

  print("Number of images: %d" % len(image_names))
  labels = np.array(
      [all_labels[os.path.splitext(n)[0][4:]] for n in image_names]).reshape(
          (-1, 1))
  image_full_paths = [os.path.join(images_path, n) for n in image_names]

  classes, cts = np.unique(list(all_labels.values()), return_counts=True)
  weight_ratio = dict(zip(classes, np.max(cts) / cts.astype(float)))
  weights = np.array([weight_ratio[l[0]] for l in labels]).reshape((-1, 1))

  loader = deepchem.data.ImageLoader()
  dat = loader.featurize(
      image_full_paths, labels=labels, weights=weights)
  if split == None:
    return dat

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter()
  }
  if not seed is None:
    np.random.seed(seed)
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dat)
  all_dataset = (train, valid, test)
  return all_dataset


def cut_raw_images(all_images, path):
  """Preprocess images:
    (1) Crop the central square including retina
    (2) Reduce resolution to 512 * 512
  """
  print("Num of images to be processed: %d" % len(all_images))
  try:
    import cv2
  except:
    logger.warn("OpenCV required for image preprocessing")
    return

  for i, img_path in enumerate(all_images):
    if i % 100 == 0:
      print("on image %d" % i)
    if os.path.join(path, 'cut_' + os.path.splitext(img_path)[0] + '.png'):
      continue
    img = cv2.imread(os.path.join(path, img_path))
    edges = cv2.Canny(img, 10, 30)
    coords = zip(*np.where(edges > 0))
    n_p = len(coords)

    coords.sort(key=lambda x: (x[0], x[1]))
    center_0 = int(
        (coords[int(0.01 * n_p)][0] + coords[int(0.99 * n_p)][0]) / 2)
    coords.sort(key=lambda x: (x[1], x[0]))
    center_1 = int(
        (coords[int(0.01 * n_p)][1] + coords[int(0.99 * n_p)][1]) / 2)

    edge_size = min(
        [center_0, img.shape[0] - center_0, center_1, img.shape[1] - center_1])
    img_cut = img[(center_0 - edge_size):(center_0 + edge_size), (
        center_1 - edge_size):(center_1 + edge_size)]
    img_cut = cv2.resize(img_cut, (512, 512))
    cv2.imwrite(
        os.path.join(path, 'cut_' + os.path.splitext(img_path)[0] + '.png'),
        img_cut)

"""
Tests for ImageLoader.
"""
from __future__ import division
from __future__ import unicode_literals

import os
import unittest
import tempfile
from scipy import misc
import deepchem as dc
import zipfile


class TestImageLoader(unittest.TestCase):
  """
  Test ImageLoader
  """

  def setUp(self):
    super(TestImageLoader, self).setUp()

    # Create image file
    self.data_dir = tempfile.mkdtemp()
    self.face = misc.face()
    self.face_path = os.path.join(self.data_dir, "face.png")
    misc.imsave(self.face_path, self.face)
    self.face_copy_path = os.path.join(self.data_dir, "face.png")
    misc.imsave(self.face_copy_path, self.face)

    # Create zip of image file
    #self.zip_path = "/home/rbharath/misc/cells.zip"
    self.zip_path = os.path.join(self.data_dir, "face.zip")
    zipf = zipfile.ZipFile(self.zip_path, "w", zipfile.ZIP_DEFLATED)
    zipf.write(self.face_path)
    zipf.close()

    # Create zip of multiple image files
    self.multi_zip_path = os.path.join(self.data_dir, "multi_face.zip")
    zipf = zipfile.ZipFile(self.multi_zip_path, "w", zipfile.ZIP_DEFLATED)
    zipf.write(self.face_path)
    zipf.write(self.face_copy_path)
    zipf.close()

  def test_simple_load(self):
    loader = dc.data.ImageLoader()
    dataset = loader.featurize(self.face_path)
    # These are the known dimensions of face.png
    assert dataset.X.shape == (1, 768, 1024, 3)

  def test_multi_load(self):
    loader = dc.data.ImageLoader()
    dataset = loader.featurize([self.face_path, self.face_copy_path])
    assert dataset.X.shape == (2, 768, 1024, 3)

  def test_zip_load(self):
    loader = dc.data.ImageLoader()
    dataset = loader.featurize(self.zip_path)
    assert dataset.X.shape == (1, 768, 1024, 3)

  def test_multi_zip_load(self):
    loader = dc.data.ImageLoader()
    dataset = loader.featurize(self.multi_zip_path)
    assert dataset.X.shape == (2, 768, 1024, 3)

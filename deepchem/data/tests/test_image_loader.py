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

    # Create zip of image files
    self.zip_path = os.path.join(self.data_dir, "face.zip")
    zipf = zipfile.ZipFile(self.zip_path, "w", zipfile.ZIP_DEFLATED)
    zipf.write(self.face_path)
    zipf.close()

  def test_simple_load(self):
    loader = dc.data.ImageLoader()
    dataset = loader.featurize(self.face_path)
    # These are the known dimensions of face.png
    assert dataset.X.shape == (1, 768, 1024, 3)

  def test_zip_load(self):
    loader = dc.data.ImageLoader()
    dataset = loader.featurize(self.zip_path)

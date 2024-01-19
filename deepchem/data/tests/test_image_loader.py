"""
Tests for ImageLoader.
"""
import os
import unittest
import tempfile
from scipy import misc
import deepchem as dc
import zipfile
import numpy as np


class TestImageLoader(unittest.TestCase):
    """
    Test ImageLoader
    """

    def setUp(self):
        super(TestImageLoader, self).setUp()
        from PIL import Image
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.tif_image_path = os.path.join(self.current_dir, "a_image.tif")

        # Create image file
        self.data_dir = tempfile.mkdtemp()
        self.face = misc.face()
        self.face_path = os.path.join(self.data_dir, "face.png")
        Image.fromarray(self.face).save(self.face_path)
        self.face_copy_path = os.path.join(self.data_dir, "face_copy.png")
        Image.fromarray(self.face).save(self.face_copy_path)

        # Create directory of multiple image files
        self.order_path = os.path.join(self.data_dir, "order_check")
        os.mkdir(self.order_path)
        self.face_path = os.path.join(self.order_path, "face.png")
        Image.fromarray(self.face).save(self.face_path)
        self.face_copy_path = os.path.join(self.order_path, "face_copy.png")
        Image.fromarray(self.face).save(self.face_copy_path)

        # Zip directory of multiple image files
        self.order_zip_path = os.path.join(self.data_dir, "order_check.zip")
        with zipfile.ZipFile(self.order_zip_path, 'w') as zip_file:
            for foldername, subfolders, filenames in os.walk(self.order_path):
                for filename in filenames:
                    file_path = os.path.join(foldername, filename)
                    arcname = os.path.relpath(file_path, self.order_path)
                    zip_file.write(file_path, arcname=arcname)

        # Create zip of image file
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

        # Create zip of multiple image files, multiple_types
        self.multitype_zip_path = os.path.join(self.data_dir,
                                               "multitype_face.zip")
        zipf = zipfile.ZipFile(self.multitype_zip_path, "w",
                               zipfile.ZIP_DEFLATED)
        zipf.write(self.face_path)
        zipf.write(self.tif_image_path)
        zipf.close()

        # Create image directory
        self.image_dir = tempfile.mkdtemp()
        face_path = os.path.join(self.image_dir, "face.png")
        Image.fromarray(self.face).save(face_path)
        face_copy_path = os.path.join(self.image_dir, "face_copy.png")
        Image.fromarray(self.face).save(face_copy_path)

    def test_png_simple_load(self):
        loader = dc.data.ImageLoader()
        dataset = loader.create_dataset(self.face_path)
        # These are the known dimensions of face.png
        assert dataset.X.shape == (1, 768, 1024, 3)

    def test_png_simple_load_with_labels(self):
        loader = dc.data.ImageLoader()
        dataset = loader.create_dataset((self.face_path, np.array(1)))
        # These are the known dimensions of face.png
        assert dataset.X.shape == (1, 768, 1024, 3)
        assert (dataset.y == np.ones((1,))).all()

    def test_png_simple_load_with_label_as_image(self):
        loader = dc.data.ImageLoader()
        dataset = loader.create_dataset((self.face_path, self.face_path))
        # These are the known dimensions of face.png
        assert dataset.X.shape == (1, 768, 1024, 3)
        assert dataset.y.shape == (1, 768, 1024, 3)

    def test_tif_simple_load(self):
        loader = dc.data.ImageLoader()
        dataset = loader.create_dataset(self.tif_image_path)
        # TODO(rbharath): Where are the color channels?
        assert dataset.X.shape == (1, 44, 330)

    def test_tif_simple_load_with_labels(self):
        loader = dc.data.ImageLoader()
        dataset = loader.create_dataset((self.tif_image_path, np.array(1)))
        # These are the known dimensions of a_image.tif
        assert dataset.X.shape == (1, 44, 330)
        assert (dataset.y == np.ones((1,))).all()

    def test_tif_simple_load_with_label_as_image(self):
        loader = dc.data.ImageLoader()
        dataset = loader.create_dataset(
            (self.tif_image_path, self.tif_image_path))
        # These are the known dimensions of a_image.tif
        assert dataset.X.shape == (1, 44, 330)
        assert dataset.y.shape == (1, 44, 330)

    def test_png_multi_load(self):
        loader = dc.data.ImageLoader()
        dataset = loader.create_dataset([self.face_path, self.face_copy_path])
        assert dataset.X.shape == (2, 768, 1024, 3)

    def test_png_zip_load(self):
        loader = dc.data.ImageLoader()
        dataset = loader.create_dataset(self.zip_path)
        assert dataset.X.shape == (1, 768, 1024, 3)

    def test_png_multi_zip_load(self):
        loader = dc.data.ImageLoader()
        dataset = loader.create_dataset(self.multi_zip_path)
        assert dataset.X.shape == (2, 768, 1024, 3)

    def test_multitype_zip_load(self):
        loader = dc.data.ImageLoader()
        dataset = loader.create_dataset(self.multitype_zip_path)
        # Since the different files have different shapes, makes an object array
        assert dataset.X.shape == (2,)

    def test_directory_load(self):
        loader = dc.data.ImageLoader()
        dataset = loader.create_dataset(self.image_dir)
        assert dataset.X.shape == (2, 768, 1024, 3)

    def test_zip_order(self):
        # Test that the order of the contents of an unzipped file is preserved.
        # Load the zip file
        loader = dc.data.ImageLoader()
        dataset_dir = loader.create_dataset(self.order_path)
        # Load multi_path directly
        loader = dc.data.ImageLoader()
        dataset_zipped = loader.create_dataset(self.order_zip_path)
        # Check that the order of the files is the same
        assert np.all(dataset_dir.X == dataset_zipped.X)

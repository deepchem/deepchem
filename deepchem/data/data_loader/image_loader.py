import os
import logging
import numpy as np
from typing import List, Optional, Union
from deepchem.data import DiskDataset, ImageDataset
from deepchem.data.data_loader.base_loader import DataLoader

logger = logging.getLogger(__name__)


class ImageLoader(DataLoader):
  """Handles loading of image files.

  This class allows for loading of images in various formats.
  For user convenience, also accepts zip-files and directories
  of images and uses some limited intelligence to attempt to
  traverse subdirectories which contain images.
  """

  def __init__(self, tasks: Optional[List[str]] = None):
    """Initialize image loader.

    At present, custom image featurizers aren't supported by this
    loader class.

    Parameters
    ----------
    tasks: list[str], optional
      List of task names for image labels.
    """
    if tasks is None:
      tasks = []
    self.tasks = tasks

  def create_dataset(
      self,
      input_files: List[str],
      labels: Optional[np.ndarray] = None,
      weights: Optional[np.ndarray] = None,
      in_memory: bool = False) -> Union[DiskDataset, ImageDataset]:
    """Creates and returns a `Dataset` object by featurizing provided image files and labels/weights.

    Parameters
    ----------
    input_files: list[str]
      Each file in this list should either be of a supported
      image format (.png, .tif only for now) or of a compressed
      folder of image files (only .zip for now).
    labels: optional
      If provided, a numpy ndarray of image labels
    weights: optional
      If provided, a numpy ndarray of image weights
    in_memory: bool
      If true, return in-memory NumpyDataset. Else return ImageDataset.

    Returns
    -------
    A `Dataset` object containing a featurized representation of data
    from `input_files`, `labels`, and `weights`.
    """
    if not isinstance(input_files, list):
      input_files = [input_files]

    image_files = []
    # Sometimes zip files contain directories within. Traverse directories
    while len(input_files) > 0:
      remainder = []
      for input_file in input_files:
        filename, extension = os.path.splitext(input_file)
        extension = extension.lower()
        # TODO(rbharath): Add support for more extensions
        if os.path.isdir(input_file):
          dirfiles = [
              os.path.join(input_file, subfile)
              for subfile in os.listdir(input_file)
          ]
          remainder += dirfiles
        elif extension == ".zip":
          zip_dir = tempfile.mkdtemp()
          zip_ref = zipfile.ZipFile(input_file, 'r')
          zip_ref.extractall(path=zip_dir)
          zip_ref.close()
          zip_files = [
              os.path.join(zip_dir, name) for name in zip_ref.namelist()
          ]
          for zip_file in zip_files:
            _, extension = os.path.splitext(zip_file)
            extension = extension.lower()
            if extension in [".png", ".tif"]:
              image_files.append(zip_file)
        elif extension in [".png", ".tif"]:
          image_files.append(input_file)
        else:
          raise ValueError("Unsupported file format")
      input_files = remainder

    if in_memory:
      return NumpyDataset(
          self.load_img(image_files), y=labels, w=weights, ids=image_files)
    else:
      return ImageDataset(image_files, y=labels, w=weights, ids=image_files)

  @staticmethod
  def load_img(image_files: List[str]) -> np.ndarray:
    from PIL import Image
    images = []
    for image_file in image_files:
      _, extension = os.path.splitext(image_file)
      extension = extension.lower()
      if extension == ".png":
        image = np.array(Image.open(image_file))
        images.append(image)
      elif extension == ".tif":
        im = Image.open(image_file)
        imarray = np.array(im)
        images.append(imarray)
      else:
        raise ValueError("Unsupported image filetype for %s" % image_file)
    return np.array(images)

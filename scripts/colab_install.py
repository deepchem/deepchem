"""
Original code by @philopon
https://gist.github.com/philopon/a75a33919d9ae41dbed5bc6a39f5ede2
"""

import sys
import os
import requests
import subprocess
import shutil
from logging import getLogger, StreamHandler, INFO

logger = getLogger(__name__)
logger.addHandler(StreamHandler())
logger.setLevel(INFO)

default_channels = [
    "-c",
    "rdkit",
    "-c",
    "conda-forge",
]
default_packages = [
    "rdkit",
]


def install(
    chunk_size=4096,
    file_name="Miniconda3-latest-Linux-x86_64.sh",
    url_base="https://repo.continuum.io/miniconda/",
    conda_path=os.path.expanduser(os.path.join("~", "miniconda")),
    add_python_path=True,
    version=None,
    # default channels are "conda-forge" and "rdkit"
    additional_channels=[],
    # default packages are "rdkit" and "deepchem"
    additional_packages=[],
    # whether to clean install or not
    force=False):
  """install deepchem on Google Colab

    For GPU/CPU notebook
    (if you don't set the version, this script will install the latest package)
    ```
    import deepchem_installer
    deepchem_installer.install(version='2.4.0')
    ```

    If you want to add soft dependent packages, you can use additional_conda_channels and 
    additional_conda_package arguments. Please see the example.
    ```
    import deepchem_installer
    deepchem_installer.install(
      version='2.4.0',
      additional_conda_channels=[]
      additional_conda_packages=["mdtraj", "networkx"]
    )

    // add channel
    import deepchem_installer
    deepchem_installer.install(
      version='2.4.0',
      additional_conda_channels=["-c", "omnia"]
      additional_conda_packages=["openmm"]
    )
    ```
  """

  python_path = os.path.join(
      conda_path,
      "lib",
      "python{0}.{1}".format(*sys.version_info),
      "site-packages",
  )

  if add_python_path and python_path not in sys.path:
    logger.info("add {} to PYTHONPATH".format(python_path))
    sys.path.append(python_path)

  if os.path.isdir(os.path.join(python_path, "deepchem")):
    logger.info("deepchem is already installed")
    if not force:
      return

    logger.info("force re-install")

  url = url_base + file_name
  python_version = "{0}.{1}.{2}".format(*sys.version_info)

  logger.info("python version: {}".format(python_version))

  if os.path.isdir(conda_path):
    logger.warning("remove current miniconda")
    shutil.rmtree(conda_path)
  elif os.path.isfile(conda_path):
    logger.warning("remove {}".format(conda_path))
    os.remove(conda_path)

  logger.info('fetching installer from {}'.format(url))
  res = requests.get(url, stream=True)
  res.raise_for_status()
  with open(file_name, 'wb') as f:
    for chunk in res.iter_content(chunk_size):
      f.write(chunk)
  logger.info('done')

  logger.info('installing miniconda to {}'.format(conda_path))
  subprocess.check_call(["bash", file_name, "-b", "-p", conda_path])
  logger.info('done')

  logger.info("installing deepchem")
  deepchem = "deepchem" if version is None else "deepchem=={}".format(version)
  subprocess.check_call([
      os.path.join(conda_path, "bin", "conda"),
      "install",
      "--yes",
      *default_channels,
      *additional_channels,
      "python=={}".format(python_version),
      *default_packages,
      *additional_packages,
      deepchem,
  ])
  logger.info("done")

  import deepchem
  logger.info("deepchem-{} installation finished!".format(deepchem.__version__))


if __name__ == "__main__":
  install()

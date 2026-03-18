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
    "conda-forge",
]
default_packages = [
    "openmm",
    "pdbfixer",
]


def install(
    chunk_size=4096,
    file_name="Miniconda3-latest-Linux-x86_64.sh",
    url_base="https://repo.continuum.io/miniconda/",
    conda_path=os.path.expanduser(os.path.join("~", "miniconda")),
    add_python_path=True,
    # default channels are "conda-forge" and "omnia"
    additional_channels=[],
    # default packages are "rdkit", "openmm" and "pdbfixer"
    additional_packages=[],
):
  """Install conda packages on Google Colab

    For GPU/CPU notebook
    ```
    import conda_installer
    conda_installer.install()
    ```

    If you want to add other packages, you can use additional_conda_channels and 
    additional_conda_package arguments. Please see the example.
    ```
    import conda_installer
    conda_installer.install(
      additional_conda_channels=[]
      additional_conda_packages=["mdtraj", "networkx"]
    )

    // add channel
    import conda_installer
    conda_installer.install(
      additional_conda_channels=["dglteam"]
      additional_conda_packages=["dgl-cuda10.1"]
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

  is_installed = []
  packages = list(set(default_packages + additional_packages))
  for package in packages:
    package = "simtk" if package == "openmm" else package
    is_installed.append(os.path.isdir(os.path.join(python_path, package)))

  if all(is_installed):
    logger.info("all packages are already installed")
    return

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

  logger.info("installing openmm, pdbfixer")
  channels = list(set(default_channels + additional_channels))
  for channel in channels:
    subprocess.check_call([
        os.path.join(conda_path, "bin", "conda"), "config", "--append",
        "channels", channel
    ])
    logger.info("added {} to channels".format(channel))
  subprocess.check_call([
      os.path.join(conda_path, "bin", "conda"),
      "install",
      "--yes",
      "python=={}".format(python_version),
      *packages,
  ])
  logger.info("done")
  logger.info("conda packages installation finished!")


if __name__ == "__main__":
  install()

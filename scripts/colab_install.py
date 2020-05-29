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


def install(
        chunk_size=4096,
        file_name="Miniconda3-latest-Linux-x86_64.sh",
        url_base="https://repo.continuum.io/miniconda/",
        conda_path=os.path.expanduser(os.path.join("~", "miniconda")),
        version=None,
        gpu=True,
        add_python_path=True,
        force=False):
    """install deepchem from miniconda on Google Colab

    For GPU notebook
    (if you don't set the version, this script will install the latest package)
    ```
    import deepchem_installer
    deepchem_installer.install()
    ```

    For CPU notebook
    ```
    import deepchem_installer
    deepchem_installer.install(version="2.3.0", gpu=False)
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
    deepchem_package = "deepchem-gpu" if gpu else "deepchem"
    subprocess.check_call([
        os.path.join(conda_path, "bin", "conda"),
        "install",
        "--yes",
        "-c", "deepchem",
        "-c", "rdkit",
        "-c", "conda-forge",
        "-c", "omnia",
        "python=={}".format(python_version),
        deepchem_package if version is None else "{}=={}".format(deepchem_package, version)])
    logger.info("done")

    import deepchem
    logger.info("deepchem-{} installation finished!".format(deepchem.__version__))


if __name__ == "__main__":
    install()

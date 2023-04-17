Notes on DeepChem CI
====================

Continuous Integration (CI) is used to continuously build and test changes made to DeepChem.
DeepChem uses Github Actions to run a number of CI jobs organized into workflows for testing the correctness of code.
When CI tests pass, it implies that the changes introduced by a commit are ready for deployment and production.
By default, on every pull request to DeepChem repository or push to the master branch of DeepChem, around 33 CI jobs are invoked across 10 workflows.
The rest of the document is organized as workflow name and the jobs in the workflow.

Build Workflow
--------------
The workflow is defined in ``build.yml`` file and runs on python version 3.9.
This workflow is only invoked on a pull request or push to the master branch and when the files in the following paths are modified:

 * "scripts/install_deepchem_conda.ps1"
 * "scripts/install_deepchem_conda.sh"
 * "requirements/\*\* "

In other words, the workflow is invoked when a new requirement is added or removed to DeepChem's requirements file or when there is a change to installation procedure scripts.

bash-build
^^^^^^^^^^
- The bash-build job is invoked macOS and ubuntu since these operating systems have ``bash`` shell.
- The job installs all DeepChem's dependencies in a miniconda environment as specified in ``scripts/install_deepchem_conda.sh`` script and performs an import check for the packages DeepChem, RdKit, PyTorch, Tensorflow and Jax.

powershell-build
^^^^^^^^^^^^^^^^
- The powershell-build job runs only in Windows OS since Windows has a powershell.
- The job installs all dependencies of a DeepChem in a miniconda environment as described in ``scripts/install_deepchem_conda.ps1`` script and performs an import check for the packages DeepChem, RdKit, PyTorch and Tensorflow. Note that an import check for ``jax`` is not performed since Jax was not supported in Windows as of Nov 2020.

Docs Workflow
-------------
The workflow is defined in ``docs.yml``.
The workflow contains one job - the ``docs-build`` job which checks whether docs get successfully built or not.

docs-build
^^^^^^^^^^
- The job runs in ubuntu latest OS with python version 3.9.
- It uses a pip cache to cache the doc build requirements. The cache key is a combination of runner OS (ubuntu) and hash of the contents in ``docs/requirements.txt`` file.
- The job installs the requirements specified in the ``docs/requirements.txt`` and checks for docs build.
- The job also performs ``doctest`` in ``docs/source/get_started/example.rst`` and ``docs/source/get_started/tutorials.rst`` file.

Formatting worflow
------------------
The workflow is defined in ``formatting.yml`` file and it performs lint checks on the source code.
To this end, it has one job - ``linting``.

linting
^^^^^^^
- Lint checks run on Ubuntu Latest OS with python version 3.9.
- Pip packages are cached using a key derived from github actions runner OS (ubuntu) and hash of the contents of files in ``requirements/`` folder.
- A micromamba environment containing all the requirements in the files env_common.yml, env_test.yml, tensorflow/env_tensorflow.cpu.yml, torch/env_torch.cpu.yml, jax/env_jax.cpu.yml is created. A combined environment provides ``mypy`` type hints for all the DeepChem modules.
- DeepChem is installed from source in the created environment.
- Yapf tests are run only in files modified in the pull request or push.
- Flake8 tests are invoked using the script ``scripts/flake8_for_ci.sh``
- Mypy type checks are performed on the DeepChem package.

Jax Workflow
------------
The workflow is defined in ``jax_setup.yml`` file.
The workflow contains two jobs - jax-build and jax-test. Both these jobs run on the latest ubuntu version with python version 3.8, 3.10. The test do not run in Windows since was not supported in Windows as of Nov 2020.

jax-build
^^^^^^^^^
Installs DeepChem with jax backend by via ``setup.py`` and tests for an import check of DeepChem and jax libraries. To install the dependencies faster, it uses pip packages cached at a key derived by runner OS and hash of file contents at ``requirements/jax/**``.

jax-test
^^^^^^^^
- The jax-tests job creates an environment file from env_common.yml, jax/env_jax.cpu.yml and env_test.yml and uses the environment file to create a miniconda environment. To speed-up install, jax-tests caches contents of `~/.cache/pip` ina key derived from runner OS and hash of file contents at ``requirements/jax/**`` folder.
- The created environment run tests which are marked as ``jax`` using pytest. The tests checks for training of neural networks by training models with jax backend on a dataset and evaluating for a suitable metric.

Mini Build Workflow
-------------------
The mini-build workflow is defined in ``mini_build.yml`` file.

core-build
^^^^^^^^^^
- The job performs minimal build of DeepChem using the core install requirements of DeepChem as defined ``setup.py`` and performs an import check in ubuntu os, python version 3.8, 3.10 and windows with python version 3.9 for deepchem package.
- The check helps to ensure that DeepChem is installable with only the minimal requirements.
- To speed up the install, it caches the folder `~/.cache/pip` at the key derived from runner OS and contents of files in ``requirements/**`` folder.

test
^^^^
- The `test` job in mini build CI runs on ubuntu os, python version 3.8, 3.10 and windows with python version 3.9 and does not perform any checks.

pypi-build
^^^^^^^^^^
This job is used for publishing nightly builds of DeepChem package from the ``master`` branch to the ``pypi`` package index. It builds deepchem nightly .whl file and uploads it to the package index.

docker-build
^^^^^^^^^^^^
The docker-build job is used for uploading latest docker images build from the tip of DeepChem master branch and uploads it to docker hub.

Release workflow
----------------
The workflow is defined in ``release.yml`` file and it is activated only when a tag is pushed.

pypi
^^^^
When a tag is pushed, the ``pypi`` job is activated which builds a .whl of DeepChem from the tip of the ``master`` branch and uploads it to pypi repository.

docker
^^^^^^
When a tag is pushed, the ``docker`` which builds a docker image corresponding to the tag and pushes the docker image to the docker hub. The ``docker`` job will only be activated when the ``pypi`` job succeeds.


Tensorflow Workflow
-------------------
The workflow is defined in ``tensorflow_setup.yml`` file.
The workflow contains two jobs - ``tf-build`` and ``tf-test``.

tf-build
^^^^^^^^
The ``tf-build`` jobs runs on ubuntu with python version 3.8 and 3.10 and windows with python version 3.9.
It install deepchem from install requirements in ``setup.py`` with ``tensorflow`` extras and performs an import check for the deepchem package.
It uses ``pip`` as an installer and caches the pip install packages at a key derived from os-name and hash of files at ``requirements/tensorflow/*``.

tf-test
^^^^^^^
The ``tf-test`` job runs on ubuntu os with python version 3.8, 3.10 and windows with python version 3.9.
The job creates a conda environment from env_common.yml, tensorflow/env_tensorflow.cpu.yml, env_test.yml files and installs deepchem on that environment via pip.
The job runs tests which are marked as `tensorflow`.
These are tests for models consisting of model training, save and restore and model evaluation on datasets with tensorflow backend.
`pip` packages are cached at a key derived from os-name and hash of files at ``requirements/tensorflow/*``.


Test Workflow
-------------
The workflow is defined in ``test.yml`` file. All the jobs in this workflow run on ubuntu os with python version 3.8, 3.10 and windows with python version 3.9.

core-build
^^^^^^^^^^
- The job performs minimal build of DeepChem using the core install requirements of DeepChem as defined ``setup.py`` and performs an import check in ubuntu os, python version 3.8, 3.10 and windows with python version 3.9 for deepchem package.
- The check helps to ensure that DeepChem is installable with only the minimal requirements.
- To speed up the install, it caches the folder `~/.cache/pip` at the key derived from runner OS and contents of files in ``requirements/**`` folder.

unit-tests
^^^^^^^^^^
- For python version 3.8, the workflow creates an environment file by combining env_common.yml env_test.yml env_ubuntu.yml tensorflow/env_tensorflow.cpu.yml torch/env_torch.cpu.yml jax/env_jax.cpu.yml env_py310_no_support.yml files. The `env_py310_no_support` file contains dependencies (vina) which are not supported in python version 3.10. For python version 3.10, an environment file is created by combining env_common.yml env_test.yml env_ubuntu.yml tensorflow/env_tensorflow.cpu.yml torch/env_torch.cpu.yml jax/env_jax.cpu.yml files.
- The dependencies are installed in conda environment and deepchem is installed in the environment via pip.
- The job runs doctest and pytest. Pytest performs unit test to check the correctness of methods and functions while doctest checks the correctness of docstrings.
- pip packages are cached based on a key derived from os name.

Torch Workflow
--------------
The workflow is defined in ``torch_setup.yml`` file.
The workflow contains two jobs - ``torch-build`` and ``torch-tests``.

torch-build
^^^^^^^^^^^
The ``torch-build`` jobs runs on ubuntu with python version 3.8 and 3.10 and windows with python version 3.9.
It install deepchem from install requirements in ``setup.py`` with ``torch`` extras and performs an import check for the deepchem package.
It uses ``pip`` as an installer and caches the pip install packages at a key derived from os-name and hash of files at ``requirements/torch/*``.

torch-tests
^^^^^^^^^^^
The ``torch-tests`` job runs on ubuntu os with python version 3.8, 3.10 and windows with python version 3.9.
The job creates a conda environment from env_common.yml, torch/env_torch.cpu.yml, env_test.yml files and installs deepchem on that environment via pip.
The job runs tests which are marked as `torch`.
These are tests for models consisting of model training, save and restore and model evaluation on datasets with torch backend.
`pip` packages are cached at a key derived from os-name and hash of file contents at ``requirements/torch/*``.


General recommendations
-----------------------

#. Handling additional or external files in unittest

When a new feature is added to DeepChem, the respective unittest should included too.
Sometimes, this test functions uses an external or additional file. To avoid problems in the CI
the absolute path of the file has to be included. For example, for the use of a file called
“Test_data_feature.csv”, the unittest function should manage the absolute path as :

::

  import os
  current_dir = os.path.dirname(os.path.abspath(__file__))
  data_dir = os.path.join(current_dir, "Test_data_feature.csv")
  result = newFeature(data_dir)

Notes on Requirement Files
--------------------------

DeepChem's CI as well as installation procedures use requirement files defined in
``requirements`` directory. Currently, there are a number of requirement files. Their
purposes are listed here.
+ `env_common.yml` - this file lists the scientific dependencies used by DeepChem like rdkit.
+ `env_ubuntu.yml` and `env_mac.yml` contain scientific dependencies which are have OS specific support. Currently, vina
+ `env_test.yml` - it is mostly used for the purpose of testing in development purpose. It contains the test dependencies.
+ The installation files in `tensorflow`, `torch` and `jax` directories contain the installation command for backend deep learning frameworks. For torch and jax, installation command is different for CPU and GPU. Hence, we use different installation files for CPU and GPU respectively.

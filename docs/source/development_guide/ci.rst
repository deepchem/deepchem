Understanding DeepChem CI
===========================

Continuous Integration(CI) is used to continuously build and run tests
for the code in your repository to make sure that the changes introduced
by the commits doesn't introduce errors. DeepChem runs a number of CI tests(jobs)
using workflows provided by Github Actions. When all CI tests in a workflow pass,
it implies that the changes introduced by a commit does not introduce any errors.

When creating a PR to master branch or when pushing to master branch, around 35 CI
tests are run from the following workflows.

Formatting CI
-------------
The workflow is defined in ``formatting.yml`` file and it performs lint checks - checks for mypy, yapf and flake8 errors. The CI runs on ubuntu-latest with python version 3.9.

- Create env.yml by combining all the requirements files - env_common.yml, env_test.yml, tensorflow/env_tensorflow.cpu.yml, torch/env_torch.cpu.yml, jax/env_jax.cpu.yml
- Install all dependencies using micrombamba (do the dependencies comes from cache)?
- Install deepchem
- Run yapf on modified files
- Run flake8 based on configuration in ``scripts/flake8_for_ci.sh``
- Run mypy test
- Cache pip packages for linux.
- The cache key is determined by the runner os and the content of `requirements` folder.
- A cache miss will occur if requirement file changes because cache key hash depends on the content of `requirements` folder.

Unit tests
----------
The workflow is defined in ``test.yml`` file.

- The CI runs on both push and pull requests to deepchem master branch.
- Builds and install DeepChem via `pip install -e .` in latest Ubuntu OS and Python 3.8, 3.10, windows 3.9 and performs deepchem import check.
- It creates an environment file from all dependencies of deepchem by combining all the requirements files - env_common.yml, env_test.yml, tensorflow/env_tensorflow.cpu.yml, torch/env_torch.cpu.yml, jax/env_jax.cpu.yml and installs dependencies via conda.
- It runs doctest and pytest. For python 3.10, pytest on docking modules is skipped as during vina was not available in python 3.10.
- pip packages are cached for ubuntu with python 3.8 and 3.10 on the same key, windows pac, windows packages are also cached where keys depends on the os.

Docs CI
-------
The workflow is defined in ``docs.yml``. It checks for docs build in ubuntu latest with Python 3.9.

Torch CI
--------
The workflow is defined in ``torch_setup.yml`` file.

- The CI runs on both push and pull requests to deepchem master branch.
- The CI contains two jobs - torch-build and torch-tests. Both these jobs run on ubuntu with python version 3.8, 3.10 and windows with python version 3.9.
- The torch-build performs an import check of pytorch and deepchem.
- The torch-tests job creates an environment from env_common.yml, torch/env_torch.cpu.yml, env_test.yml and runs pytests marked as torch.
- The torch-tests caches contents of `~/.cache/pip` in key which depends on os and not on python version.

Tensorflow CI
-------------
The workflow is defined in ``tensorflow_setup.yml`` file.

- The CI runs on both push and pull requests to deepchem master branch.
- The CI contains two jobs - tf-build and tf-tests. Both these jobs run on ubuntu with python version 3.8, 3.10 and windows with python version 3.9.
- The tf-build performs an import check of tensorflow and deepchem.
- The tf-tests job creates an environment from env_common.yml, tensorflow/env_tensorflow.cpu.yml, env_test.yml and runs pytests marked as tensorflow.
- The tf-tests caches contents of `~/.cache/pip` in a key which depends on os and not on python version.

Jax CI
------
The workflow is defined in ``jax_setup.yml`` file.

- The CI runs on both push and pull requests to deepchem master branch.
- The CI contains two jobs - jax-build and jax-test. Both these jobs run on ubuntu with python version 3.8, 3.10. Jax test do not run in windows.
- The jax-build performs an import check of pyjax with deepchem.
- The jax-tests job creates an environment from env_common.yml, jax/env_jax.cpu.yml, env_test.yml and runs pytests marked as jax.
- The jax-tests caches contents of `~/.cache/pip` in key which depends on os and not on python version.

Release CI
----------
The workflow is defined in ``release.yml`` file.

- The workflow is run when a tag is pushed to deepchem master branch.
- It builds deepchem whl file and uploads it to pypi repository.
- It builds deepchem docker image and uploads it to docker registry.

Build CI
--------
The workflow is defined in ``build.yml`` file.

- The workflow is run when a new requirement is added to a requirements file.
- It builds deepchem in macOS, ubuntu and windows and checks for `import deepchem`

Mini Build CI
-------------
The workflow is defined in ``mini_build.yml`` file.

- The `core-build` job of this workflow makes a mini build of deepchem, with only the core requirements as defined in `setup.py` and performs an import check in ubuntu os, python version 3.8, 3.10 and windows with python version 3.9.
- The `core-build` uses cache to store requirements.
- The `test` job in minibuild CI do not perform any tests.

General recommendations

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

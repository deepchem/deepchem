Understanding DeepChem CI
===========================

Continuous Integration(CI) is used to continuously build and run tests
for the code in your repository to make sure that the changes introduced
by the commits doesn't introduce errors. DeepChem runs a number of CI tests(jobs)
using workflows provided by Github Actions. When all CI tests in a workflow pass,
it implies that the changes introduced by a commit does not introduce any errors.

When creating a PR to master branch or when pushing to master branch, around 35 CI
tests are run from the following workflows.

#. Tests for DeepChem Core - The jobs are defined in the ``.github/workflows/main.yml`` file. The following jobs are performed in this workflow:
    * Building and installation of DeepChem in latest Ubuntu OS and Python 3.8-3.11 and it checks for ``import deepchem`` 
    * These tests run on Ubuntu latest version using Python 3.8-3.11 and on windows latest version using Python 3.8. The jobs are run for checking coding conventions using yapf, flake8 and mypy. It also includes tests for doctest and code-coverage.
    * Tests for pypi-build and docker-build are also include but they are mostly skipped.

#. Tests for DeepChem Common - The jobs are defined in the ``.github/workflows/common_setup.yml`` file. The following tests are performed in this workflow:
    * For build environments of Python 3.8, 3.9, 3.10, 3.11, DeepChem is built and import checking is performed.
    * The tests are run for checking pytest. All pytests which are not marked as jax, tensorflow or pytorch is run on ubuntu latest with Python 3.8, 3.9, 3.10, 3.11 and 3.9 and on windows latest, it is run with Python 3.9.

#. Tests for DeepChem Jax/Tensorflow/PyTorch
    * Jax - DeepChem with jax backend is installed and import check is performed for deepchem and jax. The tests for pytests with jax markers are run on ubuntu latest with Python 3.9-3.11.
    * Tensorflow - DeepChem with tensorflow backend is installed and import check is performed for DeepChem and tensorflow. The tests for pytests with tensorflow markers are run on ubuntu latest with Python 3.8-3.11 and on windows latest, it is run with Python 3.9.
    * PyTorch - DeepChem with pytorch backend is installed and import check is performed for DeepChem and torch. The tests for pytests with pytorch markers are run on ubuntu latest with Python 3.8-3.11 and on windows latest, it is run with Python 3.9.

#. Tests for documents
    * These tests are used for checking docs build. It is run on ubuntu latest with Python 3.9.

#. Tests for Release
    * These tests are run only when pushing a tag. It is run on ubuntu latest with Python 3.9.

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

Website Rebuild Trigger
-----------------------

Whenever new commits are pushed to the master branch, it triggers the `website_build_dispatch` workflow. This workflow uses `peter-evans/repository-dispatch` github action to send a repository dispatch event called `rebuild-website` to the `deepchem.github.io` repository, which will rebuild the `deepchem.io` website and the `The deepchem Book`.

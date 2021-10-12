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
    * Building and installation of DeepChem in latest Ubuntu OS and Python 3.7 and it checks for ``import deepchem`` 
    * These tests run on Ubuntu latest version using Python 3.7-3.9 and on windows latest version using Python 3.7. The jobs are run for checking coding conventions using yapf, flake8 and mypy. It also includes tests for doctest and code-coverage.
    * Tests for pypi-build and docker-build are also include but they are mostly skipped.

#. Tests for DeepChem Common - The jobs are defined in the ``.github/workflows/common_setup.yml`` file. The following tests are performed in this workflow:
    * For build environments of Python 3.7, 3.8 and 3.9, DeepChem is built and import checking is performed.
    * The tests are run for checking pytest. All pytests which are not marked as jax, tensorflow or pytorch is run on ubuntu latest with Python 3.7, 3.8 and 3.9 and on windows latest, it is run with Python 3.7.

#. Tests for DeepChem Jax/Tensorflow/PyTorch
    * Jax - DeepChem with jax backend is installed and import check is performed for deepchem and jax. The tests for pytests with jax markers are run on ubuntu latest with Python 3.7, 3.8 and 3.9.
    * Tensorflow - DeepChem with tensorflow backend is installed and import check is performed for DeepChem and tensorflow. The tests for pytests with tensorflow markers are run on ubuntu latest with Python 3.7-3.9 and on windows latest, it is run with Python 3.7.
    * PyTorch - DeepChem with pytorch backend is installed and import check is performed for DeepChem and torch. The tests for pytests with pytorch markers are run on ubuntu latest with Python 3.7-3.9 and on windows latest, it is run with Python 3.7.

#. Tests for documents
    * These tests are used for checking docs build. It is run on ubuntu latest with Python 3.7.

#. Tests for Release
    * These tests are run only when pushing a tag. It is run on ubuntu latest with Python 3.7.

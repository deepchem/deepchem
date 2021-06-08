# Release

This note explains how to release deepchem packages.

## How to release

1. Create and merge a release PR
    - Modify the version in `deepchem/__init__.py` (Remove `.dev`, e.g. `2.4.0.dev` -> `2.4.0`)
    - Update the documents for installing a new package in `README.md` and `docs`
    - Update the dockerfile at `deepchem/docker/tag/Dockerfile`
2. Push a new tag to the merge commit -> release new PyPI package and docker image
3. Create and merge a release PR in the [feedstock repository](https://github.com/conda-forge/deepchem-feedstock) -> release new Conda Forge package
4. Publish the documents for a new tag in [ReadTheDocs](https://readthedocs.org/projects/deepchem/versions/).
5. Create and merge a PR for bumping the version
    - Modify the version in `deepchem/__init__.py` again (Set the next dev version, e.g. `2.4.0` -> `2.5.0.dev`)

## PyPI

### Nightly build version

We publish nightly build packages only when merging PRs to the master.
The publish process is automated by GitHub Actions and it is in `pypi-build` section of `.github/workflows/main.yml`.

### Major version

We publish a major version package only when pushing a new tag.
The publish process is automated by GitHub Actions and it is in `pypi` section of `.github/workflows/release.yml`.

## Conda Forge

We have [the feedstock repository](https://github.com/conda-forge/deepchem-feedstock) for managing the build recipe for conda-forge.
After pushing a new tag, we create a PR for publishing a new package.
Basically, we need to modify the version of deepchem and dependencies like TensorFlow in `recipe/meta.yml`.
After merging the PR, we could publish a new package.

## Docker

### Nightly build version

The latest tag (deepchemio/deepchem:latest) is a nightly build and the image is built by `docker/nightly/Dockerfile`.
We publish nightly build images only when merging PRs to the master.
The publish process is automated by GitHub Actions and it is in `docker-build` section of `.github/workflows/main.yml`.

### Major version

We publish a major version image only when pushing a new tag.
The publish process is automated by GitHub Actions and it is in `docker` section of `.github/workflows/release.yml`.

## Docs

We should manually modify documents for installing a new package before pushing a new tag.
Basically, we modify `README.md` and `docs/get_started/installation.rst`. (include `docs/index.rst` in some cases)
If the release fixes or changes a known issue that was listed in `docs/src/issues.rst`, please update that page also.
After pushing a new tag, we go to [the project page](https://readthedocs.org/projects/deepchem/versions) in ReadTheDocs and publish the documents for a new tag.

## Website

We should manually modify the DeepChem website's installation instructions after each new stable release.
This can be done by modifying the text strings in the jQuery code at the bottom of github.com/deepchem.github.io/index.html. When the changes are pushed to github.com/deepchem/deepchem.github.io, the website will automatically update.

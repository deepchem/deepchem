# Release (WIP)

This note explains how to release deepchem packages.

## How to release

1. Create and merge a release PR (just modify the version in `deepchem/__init__.py`)
2. Push a new tag in the merge commit -> release in PyPI
3. Create and merge a release PR in the [feedstock repository](https://github.com/conda-forge/deepchem-feedstock) -> release in conda forge
4. Create and merge a PR for updating the Dockerfile (`docker/conda-forge/Dockerfile`)
5. Build and publish a new docker image -> release in DockerHub 
6. Fix the version in README.md and Documentation

## PyPI

### Nightly build

We publish a nightly build only when merging a PR to the master and passing all CI checks in a merge commit.
**If some CI check doesn't pass in a merge commit, the nightly build package will not be published.**
The publish process is automated by GitHub Actions and it is in `deploy` section of `.github/workflows/main.yml`.

### Major version build

We publish a major version build only when pushing a new tag.
The publish process is automated by GitHub Actions and it is in `.github/workflows/release.yml`.

## Conda Forge

We have [the feedstock repository](https://github.com/conda-forge/deepchem-feedstock) for managing the build recipe for conda-forge.
After pushing a new tag, we create a PR for publishing a new build.
Basically, we need to modify the version of deepchem and dependencies like TensorFlow in `recipe/meta.yml`.
After merging a PR, we could publish a new package.

## Docker

### Nightly build

The latest tag (deepchemio/deepchem:latest) is a nightly build and the image is built by `docker/master/Dockerfile`.
We publish a nightly build only when merging a PR to the master.
The publish process is automated by [Docker Hub](https://docs.docker.com/docker-hub/builds/).

### Major version build

The specific tag (deepchemio/deepchem:2.3.0) is a major version build and the image is built by `docker/conda-forge/Dockerfile`.
After publishing a new conda package, we need to modify `docker/conda-forge/Dockerfile`, build and publish a new image manually.

```bash
$ cd docker/conda-forge
$ docker build . -t deepchem:X.X.X
$ docker push deepchem:X.X.X
```

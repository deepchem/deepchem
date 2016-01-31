This is a recipe for building the current development package into a conda
binary.

The installation on travis-ci is done by building the conda package,
installing it, running the tests, and then if successful pushing the
docs to AWS S3.

# DeepChem Docs Overview

This directory contains the DeepChem docs. DeepChem's docs aim to
serve as another resource to complement our collection of tutorials
and examples.

## Building the Documentation

To build the docs, you can use the `Makefile` that's been added to
this directory. (Note that `deepchem` must be installed first.) To
generate docs in html, run

```
$ pip install -r ../requirements-docs.txt
$ make html
// clean build
$ make clean html
$ open build/html/index.html
```

If you want to confirm logs in more detail

```
$ make clean html SPHINXOPTS=-vvv
```

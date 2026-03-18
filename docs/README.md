# DeepChem Docs Overview

This directory contains the DeepChem docs. DeepChem's docs aim to
serve as another resource to complement our collection of tutorials
and examples.

## Building the Documentation

To build the docs, you can use the `Makefile` that's been added to
this directory. To generate docs in html, run following commands.

```
$ pip install -r requirements.txt
$ make html
// clean build
$ make clean html
$ open build/html/index.html
```

If you want to confirm logs in more details,

```
$ make clean html SPHINXOPTS=-vvv
```

If you want to confirm the example tests,

```
$ make doctest_examples
```
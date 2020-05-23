# DeepChem Docs Overview

This directory contains the DeepChem docs. DeepChem's docs aim to
serve as another resource to complement our collection of tutorials
and examples.

## Building the Documentation

To build the docs, you can use the `Makefile` that's been added to
this directory. (Note that `deepchem` must be installed first.) To
generate docs in html, run

```
pip install -r requirements.txt
make html
open _build/html/index.html
```

You can generate docs in other formats as well if you like. To clean up past builds run

```
make clean
```



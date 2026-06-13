Contributing your first featurizer
===================================

.. contents:: Table of contents
   :local:
   :depth: 2

Overview
--------

A **featurizer** in DeepChem transforms raw molecular inputs (SMILES
strings, RDKit molecules, or other formats) into numerical arrays that
machine learning models can consume.

This guide walks you through contributing a new featurizer to DeepChem
from scratch, including the class implementation, registration, unit
tests, and pull request process.

The class hierarchy
-------------------

All featurizers inherit from the base ``dc.feat.Featurizer`` class.
For molecules, subclass ``MolecularFeaturizer``::

    Featurizer
    └── MolecularFeaturizer       ← for SMILES / RDKit molecules
        ├── CircularFingerprint
        ├── RDKitDescriptors
        └── YourNewFeaturizer     ← you are here

Writing the featurizer
----------------------

Create a new file in ``deepchem/feat/``. The filename should be
lowercase with underscores, e.g. ``my_featurizer.py``.

**Minimal working example**::

    from typing import Optional
    import numpy as np
    from deepchem.feat import MolecularFeaturizer
    from deepchem.utils.typing import RDKitMol


    class MolecularWeightFeaturizer(MolecularFeaturizer):
        """Returns the exact molecular weight as a 1-D array.

        Examples
        --------
        >>> featurizer = MolecularWeightFeaturizer()
        >>> features = featurizer.featurize(["CCO"])
        >>> features.shape
        (1, 1)
        """

        def __init__(self) -> None:
            super().__init__()

        def _featurize(self, datapoint: RDKitMol,
                       **kwargs) -> np.ndarray:
            from rdkit.Chem import Descriptors
            mw = Descriptors.ExactMolWt(datapoint)
            return np.array([mw], dtype=np.float32)

Rules to follow
^^^^^^^^^^^^^^^

* Implement ``_featurize``, not ``featurize``.
* Soft dependencies go *inside* ``_featurize``, not at the module level.
* Always return ``np.ndarray`` with an explicit ``dtype``.
* Write NumPy-style docstrings with ``Examples``, ``Parameters``,
  ``Returns``, and ``References`` sections.

Registering your featurizer
----------------------------

Open ``deepchem/feat/__init__.py`` and add two things:

1. An import statement (in alphabetical order)::

    from deepchem.feat.molecular_weight_featurizer import (
        MolecularWeightFeaturizer
    )

2. An entry in ``__all__`` (in alphabetical order)::

    __all__ = [
        ...
        'MolecularWeightFeaturizer',
        ...
    ]

Writing unit tests
------------------

Create ``deepchem/feat/tests/test_molecular_weight_featurizer.py``::

    import pytest
    import numpy as np
    import deepchem as dc


    def test_ethanol_weight():
        f = dc.feat.MolecularWeightFeaturizer()
        result = f.featurize(["CCO"])
        assert result.shape == (1, 1)
        assert abs(result[0][0] - 46.04) < 0.01


    def test_batch_featurization():
        f = dc.feat.MolecularWeightFeaturizer()
        smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
        result = f.featurize(smiles)
        assert result.shape == (3, 1)
        assert all(result[:, 0] > 0)


    def test_invalid_smiles_returns_zero():
        f = dc.feat.MolecularWeightFeaturizer()
        result = f.featurize(["INVALID_SMILES"])
        assert result.shape == (1, 1)

Run with::

    pytest deepchem/feat/tests/test_molecular_weight_featurizer.py -v

Opening a Pull Request
-----------------------

See the :ref:`pull-request-guide` section for the complete PR
checklist and review process.

.. seealso::

   * :class:`deepchem.feat.MolecularFeaturizer`
   * :class:`deepchem.feat.RDKitDescriptors`
   * `DeepChem Featurizers API reference <https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html>`_
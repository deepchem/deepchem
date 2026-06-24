Contributing Your First Featurizer
==================================

This guide walks through the process of writing, testing, and contributing
a new featurizer to DeepChem. Featurizers are the first step in any
DeepChem pipeline: they transform raw data (molecules, crystals, sequences)
into numerical representations that machine learning models can consume.

If you are new to DeepChem development, start with :doc:`coding` for the
general coding conventions and :doc:`ci` for the CI setup.

Featurizer Class Hierarchy
--------------------------

All featurizers inherit from ``Featurizer``, an abstract base class
defined in ``deepchem/feat/base_classes.py``. The class hierarchy is:

* **Featurizer** — root base class. Subclasses must implement
  ``_featurize(self, datapoint, **kwargs)`` which processes a single
  data point. The public ``featurize()`` method handles batching, error
  logging, and array construction automatically.

  * **MolecularFeaturizer** — for small molecules. Accepts SMILES
    strings or RDKit Mol objects as input. The ``featurize()`` method
    handles SMILES-to-Mol conversion and canonical atom ordering.
    Subclasses must implement ``_featurize(self, mol, **kwargs)``.

    * Examples: ``CircularFingerprint``, ``RawFeaturizer``,
      ``RDKitDescriptors``, ``MolGraphConvFeaturizer``

  * **ComplexFeaturizer** — for molecular complexes (ligand–protein
    pairs). Expects tuples of ``(ligand_filename, protein_filename)``.

  * **MaterialStructureFeaturizer** — for inorganic crystal structures
    (Pymatgen ``Structure`` objects).

  * **MaterialCompositionFeaturizer** — for crystal compositions
    (Pymatgen ``Composition`` objects).

  * **PolymerFeaturizer** — for polymer representations (BigSMILES
    strings or weighted directed graphs).

Full Working Example: ``MolecularWeightFeaturizer``
----------------------------------------------------

We will build a ``MolecularWeightFeaturizer`` that computes the molecular
weight of a molecule using RDKit. This is the simplest possible featurizer
and is a good template for your own.

Step 1: Create the featurizer class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a new file at ``deepchem/feat/molecule_featurizers/molecular_weight_featurizer.py``:

.. code-block:: python

    from typing import Union

    from rdkit import Chem
    from deepchem.utils.typing import RDKitMol
    from deepchem.feat.base_classes import MolecularFeaturizer


    class MolecularWeightFeaturizer(MolecularFeaturizer):
        \"\"\"Calculate the molecular weight of a molecule.

        This featurizer computes the exact molecular weight for a given
        molecule using RDKit's ``Chem.Descriptors.ExactMolWt``.

        Examples
        --------
        >>> import deepchem as dc
        >>> smiles = [\"CCO\", \"CCCC\"]
        >>> featurizer = dc.feat.MolecularWeightFeaturizer()
        >>> weights = featurizer.featurize(smiles)
        >>> weights.shape
        (2, 1)

        Note
        ----
        This class requires RDKit to be installed.
        \"\"\"

        def _featurize(self, datapoint: RDKitMol, **kwargs) -> Union[float, Chem.rdchem.Mol]:
            \"\"\"Compute the molecular weight for a single molecule.

            Parameters
            ----------
            datapoint: rdkit.Chem.rdchem.Mol
                An RDKit Mol object.

            Returns
            -------
            float
                The exact molecular weight of the input molecule.
            \"\"\"
            from rdkit.Chem import Descriptors
            return Descriptors.ExactMolWt(datapoint)

Step 2: What is happening under the hood?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a user calls ``featurizer.featurize([\"CCO\", \"CCCC\"])``:

1. ``MolecularFeaturizer.featurize()`` receives the list of SMILES strings.
2. Each SMILES string is converted to an RDKit Mol object (with canonical
   atom ordering by default).
3. Your ``_featurize()`` method receives each Mol object and returns
   ``Descriptors.ExactMolWt(mol)``.
4. The parent class collects all results into a ``numpy`` array and returns it.

If a molecule cannot be parsed, the parent class logs a warning and inserts
an empty array so the pipeline does not crash on a single bad input.

Registering and Exporting the Featurizer
----------------------------------------

For users to access your featurizer as ``dc.feat.MolecularWeightFeaturizer()``,
it must be imported in the ``__init__.py`` files.

Step 1: Add to the molecule featurizers package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Open ``deepchem/feat/molecule_featurizers/__init__.py`` and add a line:

.. code-block:: python

    from deepchem.feat.molecule_featurizers.molecular_weight_featurizer import MolecularWeightFeaturizer

Step 2: Add to the top-level feat package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Open ``deepchem/feat/__init__.py`` and add a line in the
``# molecule featurizers`` section:

.. code-block:: python

    from deepchem.feat.molecule_featurizers import MolecularWeightFeaturizer

Now ``import deepchem as dc; dc.feat.MolecularWeightFeaturizer()``
will work.

Writing Unit Tests
------------------

Unit tests ensure your featurizer produces correct output and handles
edge cases gracefully. DeepChem uses ``pytest`` and ``unittest``.

Create a file at ``deepchem/feat/tests/test_molecular_weight_featurizer.py``:

.. code-block:: python

    import unittest
    import deepchem as dc
    import numpy as np


    class TestMolecularWeightFeaturizer(unittest.TestCase):

        def test_molecular_weight_single_molecule(self):
            \"\"\"Test featurizer on a single SMILES string.\"\"\"
            smiles = \"CCO\"
            featurizer = dc.feat.MolecularWeightFeaturizer()
            features = featurizer.featurize([smiles])
            assert features.shape == (1, 1)
            assert np.isclose(features[0][0], 46.0419, rtol=1e-3)

        def test_molecular_weight_multiple_molecules(self):
            \"\"\"Test featurizer on multiple SMILES strings.\"\"\"
            smiles = [\"C\", \"CC\", \"CCC\"]
            featurizer = dc.feat.MolecularWeightFeaturizer()
            features = featurizer.featurize(smiles)
            assert features.shape == (3, 1)
            expected = [16.0313, 30.0470, 44.0626]
            for i in range(len(smiles)):
                assert np.isclose(features[i][0], expected[i], rtol=1e-3)

        def test_molecular_weight_invalid_smiles(self):
            \"\"\"Test featurizer gracefully handles an invalid SMILES.\"\"\"
            smiles = [\"CCO\", \"INVALID\", \"CCCC\"]
            featurizer = dc.feat.MolecularWeightFeaturizer()
            features = featurizer.featurize(smiles)
            # The invalid SMILES should produce an empty array entry
            assert features.shape == (3,)

Run the test locally:

.. code-block:: bash

    python -m pytest deepchem/feat/tests/test_molecular_weight_featurizer.py -v

Tests follow these conventions:

* Use ``unittest.TestCase`` as the base class.
* Each test method starts with ``test_``.
* Test both normal cases and edge cases (invalid inputs, empty lists).
* Use ``numpy`` close comparisons (``np.isclose``) for floating point.
* Keep tests fast — aim for under a few seconds each.
* If a test must be slow, mark it with ``@pytest.mark.slow``.

Building Docs Locally
---------------------

To verify the documentation renders correctly, build the docs from
the repository root:

.. code-block:: bash

    cd docs
    pip install -r requirements.txt
    make html

Open ``docs/build/html/index.html`` in a browser and navigate to the
**Development Guide** section to see your new page.

If you only want to check for Sphinx build errors (without generating
full HTML), run:

.. code-block:: bash

    cd docs
    make html 2>&1 | grep -E "WARNING|ERROR"

PR Checklist
------------

Before submitting your pull request, confirm the following:

- [ ] The featurizer class is placed in the correct module
      (``molecule_featurizers/``, ``material_featurizers/``, etc.).
- [ ] ``_featurize()`` is implemented and returns the correct type.
- [ ] The class is registered in the subpackage ``__init__.py`` and
      the top-level ``deepchem/feat/__init__.py``.
- [ ] Docstrings follow the `numpy convention`_ and include a working
      ``Examples`` section with ``>>>`` doctests.
- [ ] A unit test file exists in ``deepchem/feat/tests/`` covering:
      - At least one normal-case test.
      - At least one edge case (invalid input, empty input).
- [ ] All tests pass locally (``python -m pytest <test_file> -v``).
- [ ] The docs build without warnings (``cd docs && make html``).
- [ ] Code is formatted with ``yapf``.
- [ ] Linting passes (``flake8 <modified file> --count``).
- [ ] No ``print()`` statements — use the ``logging`` module instead.
- [ ] Type annotations are present on all function signatures.

.. _numpy convention: https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard

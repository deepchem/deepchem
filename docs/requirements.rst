Requirements
------------

Hard requirements
^^^^^^^^^^^^^^^^^

DeepChem currently supports Python 3.5 through 3.7 and requires these packages on any condition.

- `joblib`_
- `NumPy`_
- `pandas`_
- `scikit-learn`_
- `SciPy`_
- `TensorFlow`_

  - `deepchem>=2.4.0` requires tensorflow v2
  - `deepchem<2.4.0` requires tensorflow v1


Soft requirements
^^^^^^^^^^^^^^^^^

DeepChem has a number of "soft" requirements.

+--------------------------------+---------------+---------------------------------------------------+
| Package name                   | Version       | Location where this package is imported           |
|                                |               | (dc: deepchem)                                    |
+================================+===============+===================================================+
| `BioPython`_                   | 1.77          | :code:`dc.utlis.genomics_utils`                   |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `OpenAI Gym`_                  | Not Testing   | :code:`dc.rl`                                     |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `matminer`_                    | 0.6.3         | :code:`dc.feat.materials_featurizers`             |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `MDTraj`_                      | 1.9.4         | :code:`dc.utils.pdbqt_utils`                      |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `NetworkX`_                    | 2.2           | :code:`dc.utils.rdkit_utils`                      |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `OpenMM`_                      | 7.4.2         | :code:`dc.utils.rdkit_utils`                      |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `PDBFixer`_                    | 1.6           | :code:`dc.utils.rdkit_utils`                      |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `Pillow`_                      | 7.1.2         | :code:`dc.data.data_loader`,                      |
|                                |               | :code:`dc.trans.transformers`                     |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `pyGPGO`_                      | 0.4.0.dev1    | :code:`dc.hyper.gaussian_process`                 |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `Pymatgen`_                    | 2020.7.3      | :code:`dc.feat.materials_featurizers`             |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `PyTorch`_                     | Not Testing   | :code:`dc.data.datasets`                          |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `PyTorch Geometric`_           | Not Testing   | :code:`dc.utils.molecule_graph`                   |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `RDKit`_                       | 2020.03.4     | Many modules                                      |
|                                |               | (we recommend you to instal)                      |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `simdna`_                      | 0.4.3.2       | :code:`dc.metrics.genomic_metrics`,               |
|                                |               | :code:`dc.molnet.dnasim`                          |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `Tensorflow Probability`_      | 0.10          | :code:`dc.rl`                                     |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `XGBoost`_                     | 0.90          | :code:`dc.models.xgboost_models`                  |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `Weights & Biases`_            | Not Testing   | :code:`dc.models.keras_model`,                    |
|                                |               | :code:`dc.models.callbacks`                       |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
          
.. _`joblib`: https://pypi.python.org/pypi/joblib
.. _`NumPy`: https://numpy.org/
.. _`pandas`: http://pandas.pydata.org/
.. _`scikit-learn`: https://scikit-learn.org/stable/
.. _`SciPy`: https://www.scipy.org/
.. _`TensorFlow`: https://www.tensorflow.org/
.. _`BioPython`: https://biopython.org/wiki/Documentation
.. _`OpenAI Gym`: https://gym.openai.com/
.. _`matminer`: https://hackingmaterials.lbl.gov/matminer/
.. _`MDTraj`: http://mdtraj.org/
.. _`NetworkX`: https://networkx.github.io/documentation/stable/index.html
.. _`OpenMM`: http://openmm.org/
.. _`PDBFixer`: https://github.com/pandegroup/pdbfixer
.. _`Pillow`: https://pypi.org/project/Pillow/
.. _`pyGPGO`: https://pygpgo.readthedocs.io/en/latest/
.. _`Pymatgen`: https://pymatgen.org/
.. _`PyTorch`: https://pytorch.org/
.. _`PyTorch Geometric`: https://pytorch-geometric.readthedocs.io/en/latest/
.. _`RDKit`: http://www.rdkit.org/ocs/Install.html
.. _`simdna`: https://github.com/kundajelab/simdna
.. _`Tensorflow Probability`: https://www.tensorflow.org/probability
.. _`XGBoost`: https://xgboost.readthedocs.io/en/latest/
.. _`Weights & Biases`: https://docs.wandb.com/

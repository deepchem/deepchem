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

  - `deepchem>=2.4.0` requires tensorflow v2 (2.2.0)
  - `deepchem<2.4.0` requires tensorflow v1 (>=1.14)


Soft requirements
^^^^^^^^^^^^^^^^^

DeepChem has a number of "soft" requirements.

+--------------------------------+---------------+---------------------------------------------------+
| Package name                   | Version       | Location where this package is imported           |
|                                |               | (dc: deepchem)                                    |
+================================+===============+===================================================+
| `BioPython`_                   | latest        | :code:`dc.utlis.genomics_utils`                   |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `Deep Graph Library`_          | latset        | :code:`dc.feat.graph_data`                        |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `OpenAI Gym`_                  | Not Testing   | :code:`dc.rl`                                     |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `matminer`_                    | latest        | :code:`dc.feat.materials_featurizers`             |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `MDTraj`_                      | latest        | :code:`dc.utils.pdbqt_utils`                      |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `Mol2vec`_                     | latest        | :code:`dc.utils.molecule_featurizers`             |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `Mordred`_                     | latest        | :code:`dc.utils.molecule_featurizers`             |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `NetworkX`_                    | latest        | :code:`dc.utils.rdkit_utils`                      |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `OpenMM`_                      | latest        | :code:`dc.utils.rdkit_utils`                      |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `PDBFixer`_                    | latest        | :code:`dc.utils.rdkit_utils`                      |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `Pillow`_                      | latest        | :code:`dc.data.data_loader`,                      |
|                                |               | :code:`dc.trans.transformers`                     |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `pyGPGO`_                      | latest        | :code:`dc.hyper.gaussian_process`                 |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `Pymatgen`_                    | latest        | :code:`dc.feat.materials_featurizers`             |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `PyTorch`_                     | 1.6.0         | :code:`dc.data.datasets`                          |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `PyTorch Geometric`_           | latest (with  | :code:`dc.feat.graph_data`                        |
|                                | PyTorch 1.6.0)|                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `RDKit`_                       | latest        | Many modules                                      |
|                                |               | (we recommend you to instal)                      |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `simdna`_                      | latest        | :code:`dc.metrics.genomic_metrics`,               |
|                                |               | :code:`dc.molnet.dnasim`                          |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `Tensorflow Probability`_      | 0.10.1        | :code:`dc.rl`                                     |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `XGBoost`_                     | latest        | :code:`dc.models.xgboost_models`                  |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `Weights & Biases`_            | Not Testing   | :code:`dc.models.keras_model`,                    |
|                                |               | :code:`dc.models.callbacks`                       |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
| `HuggingFace Transformers`_    | Not Testing   | :code:`dc.feat.smiles_tokenizer`                  |
|                                |               |                                                   |
|                                |               |                                                   |
+--------------------------------+---------------+---------------------------------------------------+
          
.. _`joblib`: https://pypi.python.org/pypi/joblib
.. _`NumPy`: https://numpy.org/
.. _`pandas`: http://pandas.pydata.org/
.. _`scikit-learn`: https://scikit-learn.org/stable/
.. _`SciPy`: https://www.scipy.org/
.. _`TensorFlow`: https://www.tensorflow.org/
.. _`BioPython`: https://biopython.org/wiki/Documentation
.. _`Deep Graph Library`: https://www.dgl.ai/
.. _`OpenAI Gym`: https://gym.openai.com/
.. _`matminer`: https://hackingmaterials.lbl.gov/matminer/
.. _`MDTraj`: http://mdtraj.org/
.. _`Mol2vec`: https://github.com/samoturk/mol2vec
.. _`Mordred`: http://mordred-descriptor.github.io/documentation/master/
.. _`NetworkX`: https://networkx.github.io/documentation/stable/index.html
.. _`OpenMM`: http://openmm.org/
.. _`PDBFixer`: https://github.com/pandegroup/pdbfixer
.. _`Pillow`: https://pypi.org/project/Pillow/
.. _`pyGPGO`: https://pygpgo.readthedocs.io/en/latest/
.. _`Pymatgen`: https://pymatgen.org/
.. _`PyTorch`: https://pytorch.org/
.. _`PyTorch Geometric`: https://pytorch-geometric.readthedocs.io/en/latest/
.. _`RDKit`: http://www.rdkit.org/docs/Install.html
.. _`simdna`: https://github.com/kundajelab/simdna
.. _`Tensorflow Probability`: https://www.tensorflow.org/probability
.. _`XGBoost`: https://xgboost.readthedocs.io/en/latest/
.. _`Weights & Biases`: https://docs.wandb.com/
.. _`HuggingFace Transformers`: https://huggingface.co/transformers/


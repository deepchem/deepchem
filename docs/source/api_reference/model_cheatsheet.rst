Model Cheatsheet
----------------
If you're just getting started with DeepChem, you're probably interested in the
basics. The place to get started is this "model cheatsheet" that lists various
types of custom DeepChem models. Note that some wrappers like :code:`SklearnModel`
and :code:`GBDTModel` which wrap external machine learning libraries are excluded,
but this table should otherwise be complete.

As a note about how to read these tables: Each row describes what's needed to
invoke a given model. Some models must be applied with given :code:`Transformer` or
:code:`Featurizer` objects. Most models can be trained calling :code:`model.fit`,
otherwise the name of the fit_method is given in the Comment column.
In order to run the models, make sure that the backend (Keras and tensorflow
or Pytorch or Jax) is installed.
You can thus read off what's needed to train the model from the table below.

**General purpose**

.. csv-table:: General purpose models
    :file: ./general_purpose_models.csv
    :width: 100%
    :header-rows: 1

**Molecules**

Many models implemented in DeepChem were designed for small to medium-sized organic molecules,
most often drug-like compounds.
If your data is very different (e.g. molecules contain 'exotic' elements not present in the original dataset)
or cannot be represented well using SMILES (e.g. metal complexes, crystals), some adaptations to the
featurization and/or model might be needed to get reasonable results.

.. csv-table:: Molecular models
    :file: ./molecular_models.csv
    :width: 100%
    :header-rows: 1

**Materials**

The following models were designed specifically for (inorganic) materials.

.. csv-table:: Material models
    :file: ./material_models.csv
    :width: 100%
    :header-rows: 1



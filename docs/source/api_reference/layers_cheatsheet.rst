Layers Cheatsheet
-----------------
DeepChem has an extensive collection of scientifically relevant differentiable layers.
The "layers cheatsheet" lists various types of custom DeepChem layers which are
Note that some layers implemented for specific models such as :code:`SklearnModel`
and :code:`GBDTModel` which are excluded,
but this table should otherwise be complete.

As a note about how to read these tables: Each row describes what's needed to
invoke a given model. Some models must be applied with given :code:`Transformer` or
:code:`Featurizer` objects. Most models can be trained calling :code:`model.fit`,
otherwise the name of the fit_method is given in the Comment column.
In order to use the layers, make sure that the backend (Keras and tensorflow
or Pytorch) is installed.
You can thus read off what's needed to train the model from the table below.

**Tensorflow Keras Layers**
These layers are subclasses of the :code:`tensorflow.keras.layers.Layer` class.
.. csv-table:: Custom Keras Layers
    :file: ./keras_layers.csv
    :width: 100%
    :header-rows: 1

**PyTorch **

These layers are subclasses of the :code:`torch.nn.Module` class.


.. csv-table:: Custom PyTorch Layers
    :file: ./torch_layers.csv
    :width: 100%
    :header-rows: 1




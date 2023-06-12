Layers Cheatsheet
-----------------
The "layers cheatsheet" lists various scientifically relevant differentiable layers implemented in DeepChem.

Note that some layers implemented for specific model architectures such as :code:`GROVER`
and :code:`Attention` layers, this is indicated in the `Model` column of the table.

In order to use the layers, make sure that the backend (Keras and tensorflow, Pytorch or Jax) is installed.

**Tensorflow Keras Layers**

These layers are subclasses of the :code:`tensorflow.keras.layers.Layer` class.

.. csv-table:: Custom Keras Layers
    :file: ./keras_layers.csv
    :width: 100%
    :header-rows: 1

**PyTorch**

These layers are subclasses of the :code:`torch.nn.Module` class.

.. csv-table:: Custom PyTorch Layers
    :file: ./torch_layers.csv
    :width: 100%
    :header-rows: 1




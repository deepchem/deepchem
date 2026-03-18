import deepchem as dc
import numpy as np
import tensorflow as tf
from deepchem.models.losses import L2Loss
from tensorflow.keras.layers import Input, Dense


class MLP(dc.models.KerasModel):

  def __init__(self, n_tasks=1, feature_dim=100, hidden_layer_size=64,
               **kwargs):
    self.feature_dim = feature_dim
    self.hidden_layer_size = hidden_layer_size
    self.n_tasks = n_tasks

    model, loss, output_types = self._build_graph()
    super(MLP, self).__init__(
        model=model, loss=loss, output_types=output_types, **kwargs)

  def _build_graph(self):
    inputs = Input(dtype=tf.float32, shape=(self.feature_dim,), name="Input")
    out1 = Dense(units=self.hidden_layer_size, activation='relu')(inputs)

    final = Dense(units=self.n_tasks, activation='sigmoid')(out1)
    outputs = [final]
    output_types = ['prediction']
    loss = dc.models.losses.BinaryCrossEntropy()

    model = tf.keras.Model(inputs=[inputs], outputs=outputs)
    return model, loss, output_types


X_1 = np.random.randn(100, 32)
y_1 = np.random.randn(100, 100)

dataset_1 = dc.data.NumpyDataset(X_1, y_1)

X_2 = np.random.randn(100, 32)
y_2 = np.random.randn(100, 10)

dataset_2 = dc.data.NumpyDataset(X_2, y_2)

source_model = MLP(feature_dim=32, hidden_layer_size=100, n_tasks=100)
source_model.fit(dataset_1, nb_epoch=100)

dest_model = MLP(feature_dim=32, hidden_layer_size=100, n_tasks=10)
dest_model.load_from_pretrained(
    source_model=source_model,
    assignment_map=None,
    value_map=None,
    model_dir=None,
    include_top=False)

dest_model.fit(dataset_2, nb_epoch=100)

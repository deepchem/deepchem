import deepchem as dc
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense


class MLP(dc.models.KerasModel):
    """
    Multi-Layer Perceptron (MLP) model for DeepChem.
    """

    def __init__(self, n_tasks=1, feature_dim=100, hidden_layer_size=64, **kwargs):
        """
        Initialize the MLP model.

        Args:
        - n_tasks (int): Number of tasks (output dimensions).
        - feature_dim (int): Dimensionality of input features.
        - hidden_layer_size (int): Number of neurons in the hidden layer.
        - **kwargs: Additional keyword arguments.

        """
        self.feature_dim = feature_dim
        self.hidden_layer_size = hidden_layer_size
        self.n_tasks = n_tasks

        # Build the computational graph for the model
        model, loss, output_types = self._build_graph()
        super(MLP, self).__init__(
            model=model, loss=loss, output_types=output_types, **kwargs)

    def _build_graph(self):
        """
        Build the computational graph for the MLP model.
        """
        # Define input layer
        inputs = Input(dtype=tf.float32, shape=(self.feature_dim,), name="Input")

        # Define hidden layer with ReLU activation
        out1 = Dense(units=self.hidden_layer_size, activation='relu')(inputs)

        # Define output layer with sigmoid activation for binary classification
        final = Dense(units=self.n_tasks, activation='sigmoid')(out1)
        outputs = [final]
        output_types = ['prediction']

        # Define the loss function as Binary Cross Entropy
        loss = dc.models.losses.BinaryCrossEntropy()

        # Create the Keras model
        model = tf.keras.Model(inputs=[inputs], outputs=outputs)
        return model, loss, output_types


# Generating random datasets
X_1 = np.random.randn(100, 32)
y_1 = np.random.randn(100, 100)
dataset_1 = dc.data.NumpyDataset(X_1, y_1)

X_2 = np.random.randn(100, 32)
y_2 = np.random.randn(100, 10)
dataset_2 = dc.data.NumpyDataset(X_2, y_2)

# Define shared parameters for source and destination models
feature_dim = 32
hidden_layer_size = 100

# Creating and training the source model on dataset_1
source_model = MLP(feature_dim=feature_dim, hidden_layer_size=hidden_layer_size, n_tasks=100)
source_model.fit(dataset_1, nb_epoch=100)

# Creating the destination model with the same architecture as the source model
dest_model = MLP(feature_dim=feature_dim, hidden_layer_size=hidden_layer_size, n_tasks=10)

# Loading weights from the pre-trained source_model to the dest_model
dest_model.load_from_pretrained(
    source_model=source_model,
    assignment_map=None,
    value_map=None,
    model_dir=None,
    include_top=False)

# Training the destination model on dataset_2
dest_model.fit(dataset_2, nb_epoch=100)

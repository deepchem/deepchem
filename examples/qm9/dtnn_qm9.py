import logging
import numpy as np

# Add logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

# Load the QM9 dataset
data = pd.read_csv('qm9.csv')

# Define the features and targets
features = ['atom_count', 'atom_type', 'bond_count', 'bond_type']
targets = ['energy', 'dipole', 'dipole_magnitude', 'dipole_direction']

# Split the data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2)

# Define the model
model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Add data validation
logger.info(f"Train set size: {len(train_data)}")
logger.info(f"Test set size: {len(test_data)}")
logger.info(f"Feature shape: {train_data.shape}")
logger.info(f"Target shape: {targets}")

# Check for NaN values
assert not np.isnan(train_data).any(), "NaN values in training features"
assert not np.isnan(targets).any(), "NaN values in training targets"

# Train the model
model.fit(train_data, epochs=10)

# Add learning curve tracking
train_scores = []
test_scores = []

for epoch in range(10):
    model.fit(train_data, epochs=1)
    
    train_score = model.evaluate(train_data, metrics=metric)
    test_score = model.evaluate(test_data, metrics=metric)
    
    train_scores.append(train_score)
    test_scores.append(test_score)
    
    logger.info(f"Epoch {epoch}: Train R²={train_score}, Test R²={test_score}")

# Add final diagnostics
logger.info(f"Final Train R²: {train_scores[-1]}")
logger.info(f"Final Test R²: {test_scores[-1]}")

if test_scores[-1] < 0.5:
    logger.warning("Poor test performance detected. Check:")
    logger.warning("1. Hyperparameters (learning rate, hidden units)")
    logger.warning("2. Data preprocessing/featurization")
    logger.warning("3. Model architecture compatibility")
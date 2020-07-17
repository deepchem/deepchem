import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.random.set_seed(123)
import deepchem as dc

# Load QM8 dataset
tasks, datasets, transformers = dc.molnet.load_qm8()
train_dataset, valid_dataset, test_dataset = datasets

# Fit models
metric = [dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")]

# Batch size of models
batch_size = 50
n_embedding = 20
n_distance = 51
distance_min = -1.
distance_max = 9.2
n_hidden = 15
model = dc.models.DTNNModel(
    len(tasks),
    n_embedding=n_embedding,
    n_hidden=n_hidden,
    n_distance=n_distance,
    distance_min=distance_min,
    distance_max=distance_max,
    output_activation=False,
    batch_size=batch_size,
    learning_rate=0.0001,
    use_queue=False,
    mode="regression")

# Fit trained model
model.fit(train_dataset, nb_epoch=50)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, metric, transformers)
valid_scores = model.evaluate(valid_dataset, metric, transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

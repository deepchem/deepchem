from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)
import deepchem as dc

# Load Tox21 dataset
tasks, datasets, transformers = dc.molnet.load_qm8()
train_dataset, valid_dataset, test_dataset = datasets

# Fit models
metric = [dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")]

# Batch size of models
batch_size = 50
n_embedding = 20
graph_model = dc.nn.SequentialDTNNGraph(n_distance=100)
graph_model.add(dc.nn.DTNNEmbedding(n_embedding=n_embedding))
graph_model.add(dc.nn.DTNNStep(n_embedding=n_embedding, n_distance=100))
graph_model.add(dc.nn.DTNNStep(n_embedding=n_embedding, n_distance=100))
graph_model.add(dc.nn.DTNNGather(n_embedding=n_embedding))
n_feat = n_embedding

model = dc.models.MultitaskGraphRegressor(
    graph_model,
    len(tasks),
    n_feat,
    batch_size=batch_size,
    learning_rate=0.0005,
    learning_rate_decay_time=1000,
    optimizer_type="adam",
    beta1=.9,
    beta2=.999)

# Fit trained model
model.fit(train_dataset, nb_epoch=50)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, metric, transformers)
valid_scores = model.evaluate(valid_dataset, metric, transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

import deepchem as dc
import numpy as np
from sklearn.ensemble import RandomForestClassifier

N = 10
n_feat = 5
n_classes = 3
n_tasks = 1
X = np.random.rand(N, n_feat)
y = np.random.randint(3, size=(N, n_tasks))
dataset = dc.data.NumpyDataset(X, y)

sklearn_model = RandomForestClassifier(
    class_weight="balanced", n_estimators=50)
model = dc.models.SklearnModel(sklearn_model)

# Fit models
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)

# Fit trained model
print("About to fit model")
model.fit(dataset)
model.save()

print("About to evaluate model")
train_scores = model.evaluate(dataset, [metric], [])

print("Train scores")
print(train_scores)

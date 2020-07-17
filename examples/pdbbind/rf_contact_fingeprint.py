import logging
import deepchem as dc
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(level=logging.INFO)

featurizer = dc.feat.ContactCircularFingerprint()
tasks, datasets, transformers = dc.molnet.load_pdbbind(subset="core", featurizer=featurizer)

train_dataset, valid_dataset, test_dataset = datasets 

metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

sklearn_model = RandomForestRegressor(n_estimators=500)
model = dc.models.SklearnModel(sklearn_model)

# Fit trained model
print("Fitting model on train dataset")
model.fit(train_dataset)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

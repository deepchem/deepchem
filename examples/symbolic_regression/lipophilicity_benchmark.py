import sys, os
sys.path.insert(0, os.path.abspath('.'))
import deepchem as dc
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from deepchem.models.symbolic_regression import GeneticProgramming, compute_descriptors

tasks, datasets, transformers = dc.molnet.load_lipo(featurizer='ECFP',reload=False)
train, val, test = datasets

X_train = compute_descriptors(train.ids)
X_test = compute_descriptors(test.ids)
y_train = train.y.flatten()
y_test = test.y.flatten()

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)

gp = GeneticProgramming(population_size=100, tournament_size=3,
                         mutation_rate=0.2, crossover_rate=0.8, task="regression")

print("Running GP on Lipophilicity (50 generations)...")
best, history = gp.evolve(X_train, y_train, generations=50, verbose=True)

preds = best.evaluate(X_test).detach().numpy()
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"\n{'='*40}")
print(f"Lipophilicity RMSE: {rmse:.4f}")
print(f"Equation:           {best}")
print(f"{'='*40}")
import sys, os
sys.path.insert(0, os.path.abspath('.'))
import deepchem as dc
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from deepchem.models.symbolic_regression import GeneticProgramming, compute_descriptors

print("Loading BACE dataset...")
tasks, datasets, transformers = dc.molnet.load_bace_classification(featurizer='ECFP')
train, val, test = datasets

print("Computing descriptors...")
X_train = compute_descriptors(train.ids)
X_test = compute_descriptors(test.ids)
y_train = train.y.flatten()
y_test = test.y.flatten()

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)

gp = GeneticProgramming(population_size=100, tournament_size=3,
                         mutation_rate=0.2, crossover_rate=0.8, task="classification")

print("\nRunning GP on BACE (50 generations)...")
best, history = gp.evolve(X_train, y_train, generations=50, verbose=True)

preds = best.evaluate(X_test).detach().numpy()
probs = 1 / (1 + np.exp(-np.clip(preds, -500, 500)))
auc   = roc_auc_score(y_test, probs)

print(f"\n{'='*40}")
print(f"BACE AUC:  {auc:.4f}")
print(f"Equation:  {best}")
print(f"{'='*40}")
if auc > 0.55:
    print("BACE benchmark PASSED!")
else:
    print("Below threshold.")



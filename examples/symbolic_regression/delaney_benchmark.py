"""
Symbolic Regression Benchmark on Delaney Dataset
"""

import sys
sys.path.insert(0, '../..')

import deepchem as dc
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

print("="*80)
print("SYMBOLIC REGRESSION ON MOLECULENET DELANEY")
print("="*80)

# Load dataset
print("\nLoading Delaney (ESOL) dataset...")
tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='ECFP')
train, val, test = datasets

print(f"  Train: {len(train)} molecules")
print(f"  Test:  {len(test)} molecules")

# Import from deepchem
from deepchem.models.symbolic_regression import GeneticProgramming, compute_descriptors

# Compute descriptors
print("\nComputing molecular descriptors...")
X_train = compute_descriptors(train.ids)
X_test = compute_descriptors(test.ids)
y_train = train.y.flatten()
y_test = test.y.flatten()

print(f"  Features: {X_train.shape}")

# Convert to torch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)

# Run Symbolic Regression
print("\n" + "="*80)
print("1. SYMBOLIC REGRESSION (Genetic Programming)")
print("="*80)

gp = GeneticProgramming(
    population_size=100,
    tournament_size=3,
    mutation_rate=0.2,
    crossover_rate=0.8
)

print("\nTraining genetic algorithm (50 generations)...")
best, history = gp.evolve(X_train_torch, y_train_torch, generations=50, verbose=True)

# Evaluate
y_pred_gp_torch = best.evaluate(X_test_torch)
y_pred_gp = y_pred_gp_torch.detach().numpy()
rmse_gp = np.sqrt(mean_squared_error(y_test, y_pred_gp))
r2_gp = r2_score(y_test, y_pred_gp)

print(f"\n✅ Symbolic Regression Results:")
print(f"   Equation: {best}")
print(f"   Test RMSE: {rmse_gp:.4f}")
print(f"   Test R²:   {r2_gp:.4f}")

# Ridge baseline
print("\n" + "="*80)
print("2. RIDGE REGRESSION (Baseline)")
print("="*80)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"\n✅ Ridge Results:")
print(f"   Test RMSE: {rmse_ridge:.4f}")
print(f"   Test R²:   {r2_ridge:.4f}")

# Random Forest baseline
print("\n" + "="*80)
print("3. RANDOM FOREST (Baseline)")
print("="*80)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print(f"\n✅ Random Forest Results:")
print(f"   Test RMSE: {rmse_rf:.4f}")
print(f"   Test R²:   {r2_rf:.4f}")

# Summary
print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)
print(f"\n{'Method':<25} {'RMSE':<12} {'R²':<12} {'Interpretable':<15}")
print("-"*80)
print(f"{'Symbolic Regression':<25} {rmse_gp:<12.4f} {r2_gp:<12.4f} {'✅ Yes':<15}")
print(f"{'Ridge Regression':<25} {rmse_ridge:<12.4f} {r2_ridge:<12.4f} {'⚠️  Somewhat':<15}")
print(f"{'Random Forest':<25} {rmse_rf:<12.4f} {r2_rf:<12.4f} {'❌ No':<15}")

print("\n" + "="*80)
print("BENCHMARK COMPLETE")
print("="*80)
"""
Fitness function for symbolic regression.

Author: Nandini A R
Date: March 16, 2026
GSoC 2026 - DeepChem Symbolic ML
"""

import torch
import numpy as np
from typing import Tuple
from sklearn.metrics import roc_auc_score
from deepchem.models.symbolic_regression.expression_tree import ExpressionTree


class FitnessFunction:

    def __init__(self, x_train, y_train, complexity_weight=0.001, task='regression'):
        self.x_train = x_train
        self.y_train = y_train
        self.complexity_weight = complexity_weight
        self.task = task
        self.y_std = float(y_train.std())
        if self.y_std < 1e-8:
            self.y_std = 1.0

    def evaluate(self, tree: ExpressionTree) -> Tuple[float, dict]:
        try:
            y_pred = tree.evaluate(self.x_train)
            complexity = tree.complexity()

            if self.task == 'classification':
                preds_np = y_pred.detach().numpy().flatten()
                probs = 1.0 / (1.0 + np.exp(-np.clip(preds_np, -500, 500)))
                targets_np = self.y_train.detach().numpy().flatten()
                try:
                    auc = roc_auc_score(targets_np, probs)
                except Exception:
                    auc = 0.5
                fitness = (1.0 - auc) + self.complexity_weight * complexity
                metrics = {'fitness': fitness, 'auc': auc, 'complexity': complexity}
            else:
                mse = torch.mean((y_pred - self.y_train) ** 2).item()
                normalized_mse = mse / (self.y_std ** 2)
                fitness = normalized_mse + self.complexity_weight * complexity
                metrics = {'fitness': fitness, 'mse': mse, 'complexity': complexity}

            return fitness, metrics

        except Exception as e:
            return float('inf'), {'fitness': float('inf'), 'error': str(e)}

    def evaluate_batch(self, trees: list) -> list:
        return [self.evaluate(tree) for tree in trees]

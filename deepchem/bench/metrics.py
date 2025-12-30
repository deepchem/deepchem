"""
Metrics computation for benchmarking.
"""
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                           mean_squared_error, mean_absolute_error, r2_score)

class BenchmarkMetrics:
    """Class for computing benchmark metrics."""

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray,
                 w: Optional[np.ndarray] = None):
        """Initialize metric calculator.
        
        Parameters
        ----------
        y_true: np.ndarray
            True values
        y_pred: np.ndarray
            Predicted values
        w: Optional[np.ndarray]
            Sample weights
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.w = w

        self._classification_metrics = {
            "roc_auc": self._roc_auc,
            "accuracy": self._accuracy,
            "f1": self._f1
        }

        self._regression_metrics = {
            "rmse": self._rmse,
            "mae": self._mae,
            "r2": self._r2
        }

    def compute_metrics(self, metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """Compute requested metrics.
        
        Parameters
        ----------
        metrics: Optional[List[str]]
            Names of metrics to compute. If None, computes all metrics
            based on predicted value type.

        Returns
        -------
        Dict[str, float]
            Dictionary of metric names and values
        """
        is_regression = np.issubdtype(self.y_pred.dtype, np.floating)
        
        if metrics is None:
            if is_regression:
                metrics = list(self._regression_metrics.keys())
            else:
                metrics = list(self._classification_metrics.keys())

        results = {}
        for metric in metrics:
            if metric in self._classification_metrics:
                results[metric] = self._classification_metrics[metric]()
            elif metric in self._regression_metrics:
                results[metric] = self._regression_metrics[metric]()
            else:
                raise ValueError(f"Unknown metric: {metric}")

        return results

    def _roc_auc(self) -> float:
        """Compute ROC AUC score."""
        return float(roc_auc_score(self.y_true, self.y_pred, sample_weight=self.w))

    def _accuracy(self) -> float:
        """Compute accuracy score."""
        return float(accuracy_score(self.y_true, self.y_pred > 0.5, sample_weight=self.w))

    def _f1(self) -> float:
        """Compute F1 score."""
        return float(f1_score(self.y_true, self.y_pred > 0.5, sample_weight=self.w))

    def _rmse(self) -> float:
        """Compute root mean squared error."""
        return float(np.sqrt(mean_squared_error(self.y_true, self.y_pred,
                                              sample_weight=self.w)))

    def _mae(self) -> float:
        """Compute mean absolute error."""
        return float(mean_absolute_error(self.y_true, self.y_pred,
                                       sample_weight=self.w))

    def _r2(self) -> float:
        """Compute R2 score."""
        return float(r2_score(self.y_true, self.y_pred, sample_weight=self.w))
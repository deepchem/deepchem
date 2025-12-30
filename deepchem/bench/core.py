"""
Core benchmarking functionality for DeepChem models.
"""
import logging
import time
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from deepchem.models import Model
from deepchem.data import Dataset
from deepchem.molnet import load_dataset
from deepchem.bench.metrics import BenchmarkMetrics
from deepchem.bench.models import ModelRegistry

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """Main class for running benchmarks across multiple models and datasets."""

    def __init__(self,
                 datasets: List[str],
                 models: List[str],
                 split: str = "random",
                 metrics: Optional[List[str]] = None,
                 n_jobs: int = 1,
                 device: str = "cpu"):
        """Initialize benchmark runner.
        
        Parameters
        ----------
        datasets: List[str]
            Names of MoleculeNet datasets to benchmark
        models: List[str]
            Names of models to evaluate
        split: str
            Dataset splitting method ("random", "scaffold", "temporal")
        metrics: Optional[List[str]]
            Metrics to compute. If None, uses default metrics for task type
        n_jobs: int
            Number of parallel jobs
        device: str
            Device to run on ("cpu" or "cuda")
        """
        self.datasets = datasets
        self.models = models
        self.split = split
        self.metrics = metrics
        self.n_jobs = n_jobs
        self.device = device
        
        self.model_registry = ModelRegistry()
        self.results = {}
        
    def run(self) -> Dict:
        """Run full benchmark suite."""
        for dataset in self.datasets:
            logger.info(f"Loading dataset: {dataset}")
            tasks, datasets, transformers = load_dataset(
                dataset, split=self.split)
            train_dataset, valid_dataset, test_dataset = datasets
            
            self.results[dataset] = {}
            
            for model_name in self.models:
                logger.info(f"Evaluating {model_name} on {dataset}")
                
                # Get model class and parameters
                model_cls, model_params = self.model_registry.get_model(
                    model_name)
                
                # Initialize model
                model = model_cls(**model_params)
                
                # Train and evaluate
                start_time = time.time()
                model.fit(train_dataset)
                train_time = time.time() - start_time
                
                # Get predictions
                y_pred = model.predict(test_dataset)
                
                # Compute metrics
                metrics = BenchmarkMetrics(
                    test_dataset.y, y_pred, valid_dataset.w)
                scores = metrics.compute_metrics(self.metrics)
                
                self.results[dataset][model_name] = {
                    "scores": scores,
                    "train_time": train_time
                }
                
        return self.results

    def get_summary(self) -> pd.DataFrame:
        """Generate summary DataFrame of results."""
        if not self.results:
            raise ValueError("No benchmark results available. Run benchmarks first.")
            
        rows = []
        for dataset, model_results in self.results.items():
            for model, results in model_results.items():
                row = {
                    "Dataset": dataset,
                    "Model": model,
                    "Train Time (s)": results["train_time"],
                }
                row.update(results["scores"])
                rows.append(row)
                
        return pd.DataFrame(rows)
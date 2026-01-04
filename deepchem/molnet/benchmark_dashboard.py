"""
Unified Benchmark Dashboard for DeepChem.

This module provides a unified interface for running, comparing, and visualizing
the performance of multiple DeepChem models across MoleculeNet datasets.
"""

import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np

import deepchem as dc
from deepchem.metrics import Metric

logger = logging.getLogger(__name__)


# =============================================================================
# Dataset Registry
# =============================================================================

CLASSIFICATION_DATASETS = {
    'tox21': {
        'loader': 'load_tox21',
        'description': 'Toxicity data on 12 biological targets',
        'tasks': 12,
        'default_metric': 'roc_auc_score'
    },
    'bace_c': {
        'loader': 'load_bace_classification',
        'description': 'BACE-1 inhibitor classification',
        'tasks': 1,
        'default_metric': 'roc_auc_score'
    },
    'bbbp': {
        'loader': 'load_bbbp',
        'description': 'Blood-brain barrier penetration',
        'tasks': 1,
        'default_metric': 'roc_auc_score'
    },
    'clintox': {
        'loader': 'load_clintox',
        'description': 'Clinical trial toxicity',
        'tasks': 2,
        'default_metric': 'roc_auc_score'
    },
    'hiv': {
        'loader': 'load_hiv',
        'description': 'HIV replication inhibition',
        'tasks': 1,
        'default_metric': 'roc_auc_score'
    },
    'muv': {
        'loader': 'load_muv',
        'description': 'Maximum Unbiased Validation',
        'tasks': 17,
        'default_metric': 'roc_auc_score'
    },
    'sider': {
        'loader': 'load_sider',
        'description': 'Side Effect Resource',
        'tasks': 27,
        'default_metric': 'roc_auc_score'
    },
}

REGRESSION_DATASETS = {
    'delaney': {
        'loader': 'load_delaney',
        'description': 'Aqueous solubility (ESOL)',
        'tasks': 1,
        'default_metric': 'mean_absolute_error'
    },
    'freesolv': {
        'loader': 'load_freesolv',
        'description': 'Free solvation energy (FreeSolv/SAMPL)',
        'tasks': 1,
        'default_metric': 'mean_absolute_error'
    },
    'sampl': {
        'loader': 'load_sampl',
        'description': 'Free solvation energy',
        'tasks': 1,
        'default_metric': 'mean_absolute_error'
    },
    'lipo': {
        'loader': 'load_lipo',
        'description': 'Lipophilicity',
        'tasks': 1,
        'default_metric': 'mean_absolute_error'
    },
    'qm7': {
        'loader': 'load_qm7',
        'description': 'QM7 atomization energies',
        'tasks': 1,
        'default_metric': 'mean_absolute_error'
    },
    'qm8': {
        'loader': 'load_qm8',
        'description': 'QM8 electronic properties',
        'tasks': 16,
        'default_metric': 'mean_absolute_error'
    },
    'qm9': {
        'loader': 'load_qm9',
        'description': 'QM9 molecular properties',
        'tasks': 12,
        'default_metric': 'mean_absolute_error'
    },
}

ALL_DATASETS = {**CLASSIFICATION_DATASETS, **REGRESSION_DATASETS}


# =============================================================================
# Model Registry
# =============================================================================

def _get_model_registry():
    """Get available models with their configurations."""
    registry = {}
    
    # GraphConvModel
    try:
        from deepchem.models import GraphConvModel
        registry['graphconv'] = {
            'class': GraphConvModel,
            'name': 'GraphConvModel',
            'featurizer': 'graphconv',
            'default_params': {
                'graph_conv_layers': [64, 64],
                'dense_layer_size': 128,
                'dropout': 0.0,
                'batch_size': 64,
            }
        }
    except ImportError:
        pass
    
    # GATModel
    try:
        from deepchem.models import GATModel
        registry['gat'] = {
            'class': GATModel,
            'name': 'GATModel',
            'featurizer': 'graphconv',
            'default_params': {
                'n_attention_heads': 8,
                'dropout': 0.0,
                'batch_size': 64,
            }
        }
    except ImportError:
        pass
    
    # GCNModel
    try:
        from deepchem.models import GCNModel
        registry['gcn'] = {
            'class': GCNModel,
            'name': 'GCNModel',
            'featurizer': 'graphconv',
            'default_params': {
                'graph_conv_layers': [64, 64],
                'dropout': 0.0,
                'batch_size': 64,
            }
        }
    except ImportError:
        pass
    
    # MPNNModel
    try:
        from deepchem.models import MPNNModel
        registry['mpnn'] = {
            'class': MPNNModel,
            'name': 'MPNNModel',
            'featurizer': 'weave',
            'default_params': {
                'n_hidden': 64,
                'batch_size': 64,
            }
        }
    except ImportError:
        pass
    
    # AttentiveFPModel
    try:
        from deepchem.models import AttentiveFPModel
        registry['attentivefp'] = {
            'class': AttentiveFPModel,
            'name': 'AttentiveFPModel',
            'featurizer': 'graphconv',
            'default_params': {
                'num_layers': 2,
                'num_timesteps': 2,
                'graph_feat_size': 200,
                'dropout': 0.0,
                'batch_size': 64,
            }
        }
    except ImportError:
        pass
    
    return registry


MODEL_REGISTRY = _get_model_registry()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    dataset: str
    model: str
    metric_name: str
    train_score: float
    valid_score: float
    test_score: Optional[float] = None
    training_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    datasets: List[str] = field(default_factory=lambda: ['tox21', 'delaney'])
    models: List[str] = field(default_factory=lambda: ['graphconv'])
    split: str = 'scaffold'
    n_epochs: int = 10
    seed: int = 42
    test: bool = True
    reload: bool = True
    output_dir: str = './benchmark_results'
    
    def validate(self):
        """Validate configuration."""
        for dataset in self.datasets:
            if dataset not in ALL_DATASETS:
                raise ValueError(f"Unknown dataset: {dataset}. Available: {list(ALL_DATASETS.keys())}")
        for model in self.models:
            if model not in MODEL_REGISTRY:
                raise ValueError(f"Unknown model: {model}. Available: {list(MODEL_REGISTRY.keys())}")


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """
    Unified benchmark runner for DeepChem models.
    
    This class orchestrates model training and evaluation across multiple
    MoleculeNet datasets, providing standardized comparison and reporting.
    
    Parameters
    ----------
    config : BenchmarkConfig
        Configuration for the benchmark run.
    
    Examples
    --------
    >>> config = BenchmarkConfig(
    ...     datasets=['tox21', 'delaney'],
    ...     models=['graphconv', 'gcn'],
    ...     n_epochs=10
    ... )
    >>> runner = BenchmarkRunner(config)
    >>> results = runner.run()
    >>> runner.print_report()
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.config.validate()
        self.results: List[BenchmarkResult] = []
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _get_loader(self, dataset: str):
        """Get the dataset loader function."""
        loader_name = ALL_DATASETS[dataset]['loader']
        return getattr(dc.molnet, loader_name)
    
    def _get_featurizer(self, model_name: str) -> dc.feat.Featurizer:
        """Get the appropriate featurizer for a model."""
        featurizer_name = MODEL_REGISTRY[model_name]['featurizer']
        if featurizer_name == 'graphconv':
            return dc.feat.ConvMolFeaturizer()
        elif featurizer_name == 'weave':
            return dc.feat.WeaveFeaturizer()
        elif featurizer_name == 'ecfp':
            return dc.feat.CircularFingerprint(size=1024)
        else:
            return dc.feat.ConvMolFeaturizer()
    
    def _get_metric(self, dataset: str) -> Metric:
        """Get the appropriate metric for a dataset."""
        metric_name = ALL_DATASETS[dataset]['default_metric']
        if metric_name == 'roc_auc_score':
            return Metric(dc.metrics.roc_auc_score, np.mean, mode='classification')
        elif metric_name == 'mean_absolute_error':
            return Metric(dc.metrics.mean_absolute_error, np.mean, mode='regression')
        elif metric_name == 'pearson_r2_score':
            return Metric(dc.metrics.pearson_r2_score, np.mean, mode='regression')
        else:
            return Metric(dc.metrics.roc_auc_score, np.mean)
    
    def _get_mode(self, dataset: str) -> str:
        """Get the task mode for a dataset."""
        if dataset in CLASSIFICATION_DATASETS:
            return 'classification'
        return 'regression'
    
    def _create_model(self, model_name: str, n_tasks: int, mode: str, **kwargs):
        """Create a model instance."""
        model_info = MODEL_REGISTRY[model_name]
        model_class = model_info['class']
        params = model_info['default_params'].copy()
        params.update(kwargs)
        
        return model_class(n_tasks=n_tasks, mode=mode, **params)
    
    def run_single(self, dataset: str, model_name: str) -> BenchmarkResult:
        """
        Run benchmark for a single dataset-model combination.
        
        Parameters
        ----------
        dataset : str
            Name of the dataset.
        model_name : str
            Name of the model.
        
        Returns
        -------
        BenchmarkResult
            Result of the benchmark run.
        """
        logger.info(f"Running benchmark: {model_name} on {dataset}")
        print(f"\n{'='*60}")
        print(f"Benchmarking {MODEL_REGISTRY[model_name]['name']} on {dataset}")
        print(f"{'='*60}")
        
        # Load dataset
        loader = self._get_loader(dataset)
        featurizer = self._get_featurizer(model_name)
        
        try:
            tasks, (train, valid, test), transformers = loader(
                featurizer=featurizer,
                splitter=self.config.split,
                reload=self.config.reload
            )
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset}: {e}")
            raise
        
        # Get mode and metric
        mode = self._get_mode(dataset)
        metric = self._get_metric(dataset)
        
        # Create model
        model = self._create_model(model_name, len(tasks), mode)
        
        # Train
        start_time = time.time()
        model.fit(train, nb_epoch=self.config.n_epochs)
        training_time = time.time() - start_time
        
        # Evaluate
        train_scores = model.evaluate(train, [metric], transformers)
        valid_scores = model.evaluate(valid, [metric], transformers)
        
        test_score = None
        if self.config.test:
            test_scores = model.evaluate(test, [metric], transformers)
            test_score = list(test_scores.values())[0]
        
        # Extract scores
        metric_name = ALL_DATASETS[dataset]['default_metric']
        train_score = list(train_scores.values())[0]
        valid_score = list(valid_scores.values())[0]
        
        result = BenchmarkResult(
            dataset=dataset,
            model=model_name,
            metric_name=metric_name,
            train_score=train_score,
            valid_score=valid_score,
            test_score=test_score,
            training_time=training_time,
            config={
                'n_epochs': self.config.n_epochs,
                'split': self.config.split,
                'seed': self.config.seed
            }
        )
        
        print(f"  Train: {train_score:.4f}")
        print(f"  Valid: {valid_score:.4f}")
        if test_score is not None:
            print(f"  Test:  {test_score:.4f}")
        print(f"  Time:  {training_time:.2f}s")
        
        return result
    
    def run(self) -> List[BenchmarkResult]:
        """
        Run all benchmarks specified in the configuration.
        
        Returns
        -------
        List[BenchmarkResult]
            List of all benchmark results.
        """
        self.results = []
        
        for dataset in self.config.datasets:
            for model in self.config.models:
                try:
                    result = self.run_single(dataset, model)
                    self.results.append(result)
                except Exception as e:
                    logger.error(f"Benchmark failed for {model} on {dataset}: {e}")
                    print(f"  ERROR: {e}")
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _save_results(self):
        """Save results to JSON file."""
        output_file = os.path.join(
            self.config.output_dir,
            f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        results_dict = {
            'config': asdict(self.config),
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    def print_report(self):
        """Print a formatted benchmark report."""
        if not self.results:
            print("No results to report. Run benchmarks first.")
            return
        
        print("\n" + "="*70)
        print(" DeepChem Benchmark Report")
        print("="*70)
        print(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Datasets:  {', '.join(self.config.datasets)}")
        print(f" Models:    {', '.join(self.config.models)}")
        print(f" Split:     {self.config.split}")
        print(f" Epochs:    {self.config.n_epochs}")
        print("="*70)
        
        # Group by dataset
        datasets = {}
        for r in self.results:
            if r.dataset not in datasets:
                datasets[r.dataset] = []
            datasets[r.dataset].append(r)
        
        for dataset, results in datasets.items():
            mode = self._get_mode(dataset)
            metric = ALL_DATASETS[dataset]['default_metric']
            
            print(f"\n Dataset: {dataset} ({mode})")
            print(f" Metric:  {metric}")
            print("-"*50)
            print(f" {'Model':<20} {'Train':>10} {'Valid':>10} {'Test':>10}")
            print("-"*50)
            
            for r in results:
                model_name = MODEL_REGISTRY[r.model]['name']
                test_str = f"{r.test_score:.4f}" if r.test_score else "N/A"
                print(f" {model_name:<20} {r.train_score:>10.4f} {r.valid_score:>10.4f} {test_str:>10}")
        
        print("\n" + "="*70)
    
    def get_comparison_table(self) -> Dict[str, Dict[str, float]]:
        """
        Get results as a comparison table.
        
        Returns
        -------
        Dict[str, Dict[str, float]]
            Nested dict: {dataset: {model: valid_score}}
        """
        table = {}
        for r in self.results:
            if r.dataset not in table:
                table[r.dataset] = {}
            table[r.dataset][r.model] = r.valid_score
        return table


# =============================================================================
# CLI Interface
# =============================================================================

def run_benchmark_cli():
    """Command-line interface for running benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='DeepChem Unified Benchmark Dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  deepchem-benchmark --datasets tox21 delaney --models graphconv gcn
  deepchem-benchmark --datasets tox21 --models graphconv gat mpnn --epochs 20
  deepchem-benchmark --list-datasets
  deepchem-benchmark --list-models
        """
    )
    
    parser.add_argument(
        '--datasets', '-d',
        nargs='+',
        default=['tox21', 'delaney'],
        help='Datasets to benchmark (default: tox21 delaney)'
    )
    
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        default=['graphconv'],
        help='Models to benchmark (default: graphconv)'
    )
    
    parser.add_argument(
        '--split', '-s',
        default='scaffold',
        choices=['scaffold', 'random', 'index'],
        help='Split strategy (default: scaffold)'
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='./benchmark_results',
        help='Output directory (default: ./benchmark_results)'
    )
    
    parser.add_argument(
        '--no-test',
        action='store_true',
        help='Skip test set evaluation'
    )
    
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List available datasets'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Handle list commands
    if args.list_datasets:
        print("\nAvailable Datasets:")
        print("-" * 60)
        print("\nClassification:")
        for name, info in CLASSIFICATION_DATASETS.items():
            print(f"  {name:<15} - {info['description']}")
        print("\nRegression:")
        for name, info in REGRESSION_DATASETS.items():
            print(f"  {name:<15} - {info['description']}")
        return
    
    if args.list_models:
        print("\nAvailable Models:")
        print("-" * 40)
        for name, info in MODEL_REGISTRY.items():
            print(f"  {name:<15} - {info['name']}")
        return
    
    # Run benchmarks
    config = BenchmarkConfig(
        datasets=args.datasets,
        models=args.models,
        split=args.split,
        n_epochs=args.epochs,
        seed=args.seed,
        test=not args.no_test,
        output_dir=args.output
    )
    
    runner = BenchmarkRunner(config)
    runner.run()
    runner.print_report()


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_benchmark(
    datasets: List[str] = ['tox21', 'delaney'],
    models: List[str] = ['graphconv'],
    n_epochs: int = 10,
    split: str = 'scaffold'
) -> List[BenchmarkResult]:
    """
    Quick benchmark function for interactive use.
    
    Parameters
    ----------
    datasets : List[str]
        List of dataset names.
    models : List[str]
        List of model names.
    n_epochs : int
        Number of training epochs.
    split : str
        Split strategy.
    
    Returns
    -------
    List[BenchmarkResult]
        List of benchmark results.
    
    Examples
    --------
    >>> results = quick_benchmark(
    ...     datasets=['tox21'],
    ...     models=['graphconv', 'gcn'],
    ...     n_epochs=5
    ... )
    """
    config = BenchmarkConfig(
        datasets=datasets,
        models=models,
        n_epochs=n_epochs,
        split=split
    )
    runner = BenchmarkRunner(config)
    results = runner.run()
    runner.print_report()
    return results


def list_available_datasets() -> Dict[str, Dict]:
    """List all available datasets with their information."""
    return ALL_DATASETS.copy()


def list_available_models() -> Dict[str, Dict]:
    """List all available models with their information."""
    return {k: {'name': v['name'], 'featurizer': v['featurizer']} 
            for k, v in MODEL_REGISTRY.items()}


if __name__ == '__main__':
    run_benchmark_cli()

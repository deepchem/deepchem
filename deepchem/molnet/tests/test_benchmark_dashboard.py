"""
Tests for the Unified Benchmark Dashboard.
"""

import pytest
import os
import tempfile
import json

try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        from deepchem.molnet.benchmark_dashboard import BenchmarkConfig
        
        config = BenchmarkConfig()
        assert config.datasets == ['tox21', 'delaney']
        assert config.models == ['graphconv']
        assert config.split == 'scaffold'
        assert config.n_epochs == 10
    
    def test_custom_config(self):
        """Test custom configuration."""
        from deepchem.molnet.benchmark_dashboard import BenchmarkConfig
        
        config = BenchmarkConfig(
            datasets=['bbbp', 'lipo'],
            models=['gcn'],
            n_epochs=5,
            split='random'
        )
        assert config.datasets == ['bbbp', 'lipo']
        assert config.models == ['gcn']
        assert config.n_epochs == 5
        assert config.split == 'random'
    
    def test_invalid_dataset(self):
        """Test validation with invalid dataset."""
        from deepchem.molnet.benchmark_dashboard import BenchmarkConfig
        
        config = BenchmarkConfig(datasets=['invalid_dataset'])
        with pytest.raises(ValueError, match="Unknown dataset"):
            config.validate()
    
    def test_invalid_model(self):
        """Test validation with invalid model."""
        from deepchem.molnet.benchmark_dashboard import BenchmarkConfig
        
        config = BenchmarkConfig(models=['invalid_model'])
        with pytest.raises(ValueError, match="Unknown model"):
            config.validate()


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""
    
    def test_result_creation(self):
        """Test creating a benchmark result."""
        from deepchem.molnet.benchmark_dashboard import BenchmarkResult
        
        result = BenchmarkResult(
            dataset='tox21',
            model='graphconv',
            metric_name='roc_auc_score',
            train_score=0.85,
            valid_score=0.80,
            test_score=0.78,
            training_time=120.5
        )
        
        assert result.dataset == 'tox21'
        assert result.model == 'graphconv'
        assert result.train_score == 0.85
        assert result.valid_score == 0.80
        assert result.test_score == 0.78
    
    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        from deepchem.molnet.benchmark_dashboard import BenchmarkResult
        
        result = BenchmarkResult(
            dataset='delaney',
            model='gcn',
            metric_name='mean_absolute_error',
            train_score=0.5,
            valid_score=0.6
        )
        
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict['dataset'] == 'delaney'
        assert result_dict['model'] == 'gcn'


class TestDatasetRegistry:
    """Tests for dataset registry."""
    
    def test_classification_datasets(self):
        """Test classification datasets are registered."""
        from deepchem.molnet.benchmark_dashboard import CLASSIFICATION_DATASETS
        
        assert 'tox21' in CLASSIFICATION_DATASETS
        assert 'bbbp' in CLASSIFICATION_DATASETS
        assert 'hiv' in CLASSIFICATION_DATASETS
        
        # Check structure
        assert 'loader' in CLASSIFICATION_DATASETS['tox21']
        assert 'description' in CLASSIFICATION_DATASETS['tox21']
        assert 'default_metric' in CLASSIFICATION_DATASETS['tox21']
    
    def test_regression_datasets(self):
        """Test regression datasets are registered."""
        from deepchem.molnet.benchmark_dashboard import REGRESSION_DATASETS
        
        assert 'delaney' in REGRESSION_DATASETS
        assert 'lipo' in REGRESSION_DATASETS
        assert 'qm7' in REGRESSION_DATASETS
    
    def test_all_datasets(self):
        """Test ALL_DATASETS contains both types."""
        from deepchem.molnet.benchmark_dashboard import (
            ALL_DATASETS, CLASSIFICATION_DATASETS, REGRESSION_DATASETS
        )
        
        assert len(ALL_DATASETS) == len(CLASSIFICATION_DATASETS) + len(REGRESSION_DATASETS)


class TestModelRegistry:
    """Tests for model registry."""
    
    @pytest.mark.torch
    def test_model_registry_structure(self):
        """Test model registry has correct structure."""
        from deepchem.molnet.benchmark_dashboard import MODEL_REGISTRY
        
        # At least graphconv should be available
        if 'graphconv' in MODEL_REGISTRY:
            model_info = MODEL_REGISTRY['graphconv']
            assert 'class' in model_info
            assert 'name' in model_info
            assert 'featurizer' in model_info
            assert 'default_params' in model_info


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_list_available_datasets(self):
        """Test listing available datasets."""
        from deepchem.molnet.benchmark_dashboard import list_available_datasets
        
        datasets = list_available_datasets()
        assert isinstance(datasets, dict)
        assert 'tox21' in datasets
        assert 'delaney' in datasets
    
    @pytest.mark.torch
    def test_list_available_models(self):
        """Test listing available models."""
        from deepchem.molnet.benchmark_dashboard import list_available_models
        
        models = list_available_models()
        assert isinstance(models, dict)


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""
    
    def test_runner_initialization(self):
        """Test runner initialization."""
        from deepchem.molnet.benchmark_dashboard import (
            BenchmarkRunner, BenchmarkConfig
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BenchmarkConfig(
                datasets=['tox21'],
                models=['graphconv'],
                output_dir=tmpdir
            )
            runner = BenchmarkRunner(config)
            
            assert runner.config == config
            assert runner.results == []
    
    def test_get_mode(self):
        """Test getting task mode for datasets."""
        from deepchem.molnet.benchmark_dashboard import (
            BenchmarkRunner, BenchmarkConfig
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BenchmarkConfig(output_dir=tmpdir)
            runner = BenchmarkRunner(config)
            
            assert runner._get_mode('tox21') == 'classification'
            assert runner._get_mode('delaney') == 'regression'
    
    def test_comparison_table_empty(self):
        """Test comparison table with no results."""
        from deepchem.molnet.benchmark_dashboard import (
            BenchmarkRunner, BenchmarkConfig
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BenchmarkConfig(output_dir=tmpdir)
            runner = BenchmarkRunner(config)
            
            table = runner.get_comparison_table()
            assert table == {}


@pytest.mark.torch
@pytest.mark.slow
class TestBenchmarkIntegration:
    """Integration tests for benchmark runner (slow)."""
    
    def test_quick_benchmark_smoke(self):
        """Smoke test for quick_benchmark function."""
        # This test is marked slow and should only run with --runslow
        pass  # Placeholder - actual benchmark would take too long for CI


class TestCLIArguments:
    """Tests for CLI argument parsing."""
    
    def test_cli_import(self):
        """Test CLI function can be imported."""
        from deepchem.molnet.benchmark_dashboard import run_benchmark_cli
        assert callable(run_benchmark_cli)

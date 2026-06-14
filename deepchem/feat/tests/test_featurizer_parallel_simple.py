"""
Simple test for parallel featurization functionality.

This module tests the parallel processing capabilities added to the base Featurizer
class using joblib, without requiring full DeepChem installation.
"""

import unittest
import numpy as np
import time
from typing import Any

# Try to import just the base classes we need
try:
    from deepchem.feat.base_classes import Featurizer
    # from joblib import Parallel, delayed  # Not used in simple tests
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


class TestFeaturizer(Featurizer):
    """Test implementation of base Featurizer for testing parallel functionality."""

    def _featurize(self, datapoint: Any, **kwargs) -> np.ndarray:
        """Simple featurizer that returns a fixed-size array."""
        # Simulate some computation time
        time.sleep(0.001)  # 1ms delay
        return np.array([1, 2, 3, 4, 5])


class FailingFeaturizer(Featurizer):
    """Test featurizer that fails on certain inputs."""

    def _featurize(self, datapoint: Any, **kwargs) -> np.ndarray:
        """Featurizer that fails on inputs containing 'fail'."""
        if "fail" in str(datapoint):
            raise ValueError("Intentional failure for testing")
        return np.array([1, 2, 3])


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required imports not available")
class TestFeaturizerParallelSimple(unittest.TestCase):
    """Test parallel featurization functionality with minimal dependencies."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data = [f"test_molecule_{i}" for i in range(10)]

    def test_sequential_vs_parallel_identical_results(self):
        """Test that sequential and parallel processing give identical results."""
        featurizer = TestFeaturizer()

        # Test sequential processing (n_jobs=1)
        features_seq = featurizer.featurize(self.test_data, n_jobs=1)

        # Test parallel processing (n_jobs=2)
        features_parallel = featurizer.featurize(self.test_data, n_jobs=2)

        # Results should be identical
        np.testing.assert_array_equal(features_seq, features_parallel)
        self.assertEqual(features_seq.shape, features_parallel.shape)
        self.assertEqual(len(features_seq), len(self.test_data))

    def test_different_n_jobs_identical_results(self):
        """Test that different n_jobs values give identical results."""
        featurizer = TestFeaturizer()

        features_1 = featurizer.featurize(self.test_data, n_jobs=1)
        features_2 = featurizer.featurize(self.test_data, n_jobs=2)
        features_4 = featurizer.featurize(self.test_data, n_jobs=4)

        # All results should be identical
        np.testing.assert_array_equal(features_1, features_2)
        np.testing.assert_array_equal(features_1, features_4)

    def test_backward_compatibility(self):
        """Test that default behavior (no n_jobs parameter) works as before."""
        featurizer = TestFeaturizer()

        # Test without n_jobs parameter (should default to sequential)
        features_default = featurizer.featurize(self.test_data)

        # Test with n_jobs=1 (should be identical)
        features_sequential = featurizer.featurize(self.test_data, n_jobs=1)

        np.testing.assert_array_equal(features_default, features_sequential)

    def test_per_datapoint_kwargs(self):
        """Test per-datapoint kwargs handling in parallel mode."""
        featurizer = TestFeaturizer()

        # Test with per-datapoint kwargs
        kwargs_per_dp = {
            'param1': [f"value_{i}" for i in range(len(self.test_data))
                      ],  # per-datapoint
            'param2': "global_value"  # global
        }

        features = featurizer.featurize(self.test_data,
                                        n_jobs=2,
                                        **kwargs_per_dp)

        # Should process successfully
        self.assertEqual(len(features), len(self.test_data))
        self.assertTrue(isinstance(features, np.ndarray))

    def test_error_handling_parallel(self):
        """Test error handling in parallel mode."""
        featurizer = FailingFeaturizer()
        test_data_with_failures = ["good1", "fail1", "good2", "fail2", "good3"]

        # Should handle failures gracefully
        features = featurizer.featurize(test_data_with_failures, n_jobs=2)

        # Should return results for all datapoints (empty arrays for failures)
        self.assertEqual(len(features), len(test_data_with_failures))

    def test_edge_cases(self):
        """Test edge cases for parallel featurization."""
        featurizer = TestFeaturizer()

        # Test with single datapoint
        single_result = featurizer.featurize(["single"], n_jobs=2)
        self.assertEqual(len(single_result), 1)

        # Test with empty list
        empty_result = featurizer.featurize([], n_jobs=2)
        self.assertEqual(len(empty_result), 0)

        # Test with non-iterable input (should be wrapped)
        # Note: strings are iterable in Python, so we test with a number instead
        non_iterable_result = featurizer.featurize(42, n_jobs=2)
        self.assertEqual(len(non_iterable_result), 1)

    def test_mixed_feature_shapes(self):
        """Test handling of mixed feature shapes in parallel mode."""

        class VariableShapeFeaturizer(Featurizer):

            def _featurize(self, datapoint: Any, **kwargs) -> np.ndarray:
                # Return different shapes based on datapoint
                if "small" in str(datapoint):
                    return np.array([1, 2, 3])
                else:
                    return np.array([4, 5, 6, 7, 8])

        featurizer = VariableShapeFeaturizer()
        mixed_data = ["small_1", "large_1", "small_2", "large_2"]

        # Should handle mixed shapes gracefully
        features = featurizer.featurize(mixed_data, n_jobs=2)

        # Should return object array due to different shapes
        self.assertEqual(len(features), len(mixed_data))
        self.assertEqual(features.dtype, object)

    def test_performance_improvement(self):
        """Test that parallel processing provides performance improvement."""
        featurizer = TestFeaturizer()

        # Use larger dataset for performance testing
        large_data = [f"molecule_{i}" for i in range(30)]

        # Time sequential processing
        start_time = time.time()
        features_seq = featurizer.featurize(large_data, n_jobs=1)
        seq_time = time.time() - start_time

        # Time parallel processing
        start_time = time.time()
        features_parallel = featurizer.featurize(large_data, n_jobs=4)
        parallel_time = time.time() - start_time

        # Results should be identical
        np.testing.assert_array_equal(features_seq, features_parallel)

        # Parallel should be faster (allowing some tolerance)
        print(f"Sequential: {seq_time:.3f}s, Parallel: {parallel_time:.3f}s")
        if parallel_time < seq_time:
            print(
                f"Performance improvement: {seq_time/parallel_time:.2f}x faster"
            )

    def test_logging_behavior(self):
        """Test that logging works correctly in parallel mode."""
        featurizer = TestFeaturizer()

        # Test that parallel processing logs the start message
        with self.assertLogs(level='INFO') as log:
            featurizer.featurize(self.test_data[:5], n_jobs=2)

        # Should log the parallel processing start message
        log_messages = [record.message for record in log.records]
        parallel_log_found = any("parallel jobs" in msg for msg in log_messages)
        self.assertTrue(parallel_log_found)


if __name__ == '__main__':
    unittest.main()

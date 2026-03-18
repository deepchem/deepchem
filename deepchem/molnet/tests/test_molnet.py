"""
Tests for molnet function
"""
import csv
import tempfile
import unittest

import numpy as np
import os
import pytest

import deepchem as dc
from deepchem.molnet.run_benchmark import run_benchmark
try:
    import torch  # noqa
    has_pytorch = True
except:
    has_pytorch = False


class TestMolnet(unittest.TestCase):
    """
    Test basic function of molnet
    """

    def setUp(self):
        super(TestMolnet, self).setUp()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

    @pytest.mark.slow
    @pytest.mark.tensorflow
    def test_delaney_graphconvreg(self):
        """Tests molnet benchmarking on delaney with graphconvreg."""
        datasets = ['delaney']
        model = 'graphconvreg'
        split = 'random'
        out_path = tempfile.mkdtemp()
        metric = [dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)]
        run_benchmark(datasets,
                      str(model),
                      metric=metric,
                      split=split,
                      out_path=out_path)
        with open(os.path.join(out_path, 'results.csv'), newline='\n') as f:
            reader = csv.reader(f)
            for lastrow in reader:
                pass
            assert lastrow[-4] == 'valid'
            assert float(lastrow[-3]) > 0.65
        os.remove(os.path.join(out_path, 'results.csv'))

    @pytest.mark.slow
    @pytest.mark.torch
    def test_qm7_multitask(self):
        """Tests molnet benchmarking on qm7 with multitask network."""
        datasets = ['qm7']
        model = 'tf_regression_ft'
        split = 'random'
        out_path = tempfile.mkdtemp()
        metric = [dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)]
        run_benchmark(datasets,
                      str(model),
                      metric=metric,
                      split=split,
                      out_path=out_path)
        with open(os.path.join(out_path, 'results.csv'), newline='\n') as f:
            reader = csv.reader(f)
            for lastrow in reader:
                pass
            assert lastrow[-4] == 'valid'
            # TODO For this dataset and model, the R2-scores are less than 0.3.
            # This has to be improved.
            # See: https://github.com/deepchem/deepchem/issues/2776
            assert float(lastrow[-3]) > 0.15
        os.remove(os.path.join(out_path, 'results.csv'))

    @pytest.mark.torch
    def test_clintox_multitask(self):
        """Tests molnet benchmarking on clintox with multitask network."""
        datasets = ['clintox']
        model = 'tf'
        split = 'random'
        out_path = tempfile.mkdtemp()
        metric = [dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)]
        run_benchmark(datasets,
                      str(model),
                      metric=metric,
                      split=split,
                      out_path=out_path,
                      test=True)
        with open(os.path.join(out_path, 'results.csv'), newline='\n') as f:
            reader = csv.reader(f)
            for lastrow in reader:
                pass
            assert lastrow[-4] == 'test'
            assert float(lastrow[-3]) > 0.7
        os.remove(os.path.join(out_path, 'results.csv'))

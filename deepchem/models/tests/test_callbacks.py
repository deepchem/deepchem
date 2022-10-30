import unittest
import pytest
import tempfile
import deepchem as dc
import numpy as np
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


class TestCallbacks(unittest.TestCase):

    @pytest.mark.torch
    def test_validation(self):
        """Test ValidationCallback."""
        tasks, datasets, transformers = dc.molnet.load_clintox()
        train_dataset, valid_dataset, test_dataset = datasets
        n_features = 1024
        model = dc.models.MultitaskClassifier(len(tasks),
                                              n_features,
                                              dropouts=0.5)

        # Train the model while logging the validation ROC AUC.

        metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)
        log = StringIO()
        save_dir = tempfile.mkdtemp()
        callback = dc.models.ValidationCallback(valid_dataset,
                                                30, [metric],
                                                log,
                                                save_dir=save_dir,
                                                save_on_minimum=False,
                                                transformers=transformers)
        model.fit(train_dataset, callbacks=callback)

        # Parse the log to pull out the AUC scores.

        log.seek(0)
        scores = []
        for line in log:
            score = float(line.split('=')[-1])
            scores.append(score)

        # The last reported score should match the current performance of the model.

        valid_score = model.evaluate(valid_dataset, [metric], transformers)
        self.assertAlmostEqual(valid_score['mean-roc_auc_score'],
                               scores[-1],
                               places=5)

        # The highest recorded score should match get_best_score().

        self.assertAlmostEqual(max(scores), callback.get_best_score(), places=5)

        # Reload the save model and confirm that it matches the best logged score.

        model.restore(model_dir=save_dir)
        valid_score = model.evaluate(valid_dataset, [metric], transformers)
        self.assertAlmostEqual(valid_score['mean-roc_auc_score'],
                               max(scores),
                               places=5)

        # Make sure get_best_score() still works when save_dir is not specified

        callback = dc.models.ValidationCallback(valid_dataset,
                                                30, [metric],
                                                log,
                                                save_on_minimum=False,
                                                transformers=transformers)
        model.fit(train_dataset, callbacks=callback)
        log.seek(0)
        scores = []
        for line in log:
            score = float(line.split('=')[-1])
            scores.append(score)

        self.assertTrue(abs(max(scores) - callback.get_best_score()) < 0.05)

    @pytest.mark.torch
    def test_callback_with_multiple_classes(self):
        n, x_shape, n_classes, n_tasks = 100, 10, 3, 2
        X = np.random.randn(n, x_shape)
        y = np.random.randint(low=0, high=n_classes, size=(n, n_tasks))

        train_dataset = dc.data.NumpyDataset(X=X[0:int(n * 0.8)],
                                             y=y[0:int(n * 0.8)])
        valid_dataset = dc.data.NumpyDataset(X=X[int(n * 0.8):],
                                             y=y[int(n * 0.8):])

        metric = dc.metrics.Metric(dc.metrics.prc_auc_score)
        validation = dc.models.ValidationCallback(valid_dataset,
                                                  interval=10,
                                                  metrics=[metric])

        model = dc.models.MultitaskClassifier(n_tasks=n_tasks,
                                              n_features=x_shape,
                                              n_classes=n_classes)
        model.fit(train_dataset, callbacks=[validation])
        result = model.evaluate(valid_dataset, metrics=metric, n_classes=3)
        self.assertIsNotNone(result)

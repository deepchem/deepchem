"""Regression tests for binary precision-recall input representations."""
import numpy as np
import pytest
from sklearn.metrics import auc, precision_recall_curve

from deepchem.metrics import prc_auc_score


@pytest.mark.parametrize("one_hot_labels", [False, True])
@pytest.mark.parametrize("two_column_predictions", [False, True])
def test_prc_auc_binary_representations(one_hot_labels, two_column_predictions):
    labels = np.array([0, 1, 1, 0])
    probabilities = np.array([0.1, 0.9, 0.6, 0.8])
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    expected = auc(recall, precision)
    y = np.eye(2)[labels] if one_hot_labels else labels
    y_pred = (np.column_stack(
        (1 - probabilities,
         probabilities)) if two_column_predictions else probabilities)

    assert prc_auc_score(y, y_pred) == pytest.approx(expected)

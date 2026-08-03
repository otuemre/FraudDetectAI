import numpy as np
import pandas as pd
import pytest

from src.components.evaluation import evaluate_model


class FakeModel:
    """
    A stand-in for a fitted classifier with hand-picked, known outputs —
    lets us verify evaluate_model's metric calculations exactly,
    independent of any real model's behavior.
    """

    def __init__(self, predictions, probabilities):
        self._predictions = np.array(predictions)
        self._probabilities = np.array(probabilities)

    def predict(self, X):
        return self._predictions

    def predict_proba(self, X):
        # Column 0 = P(class 0), Column 1 = P(class 1) — matches sklearn's convention
        return np.column_stack([1 - self._probabilities, self._probabilities])


def _dummy_X(n_rows: int) -> pd.DataFrame:
    """Feature values are irrelevant here since FakeModel ignores X entirely."""
    return pd.DataFrame({"v1": np.zeros(n_rows)})


def test_evaluate_model_perfect_predictions():
    y_test = pd.Series([0, 0, 1, 1])
    model = FakeModel(predictions=[0, 0, 1, 1], probabilities=[0.1, 0.2, 0.9, 0.95])

    metrics = evaluate_model(model, _dummy_X(4), y_test)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["confusion_matrix"]["true_positive"] == 2
    assert metrics["confusion_matrix"]["true_negative"] == 2
    assert metrics["confusion_matrix"]["false_positive"] == 0
    assert metrics["confusion_matrix"]["false_negative"] == 0


def test_evaluate_model_known_confusion_matrix():
    # y_true: [0, 0, 1, 1, 1] ; y_pred: [0, 1, 1, 0, 1]
    # -> TN=1 (idx0), FP=1 (idx1), TP=2 (idx2,4), FN=1 (idx3)
    y_test = pd.Series([0, 0, 1, 1, 1])
    model = FakeModel(
        predictions=[0, 1, 1, 0, 1],
        probabilities=[0.1, 0.6, 0.8, 0.3, 0.9],
    )

    metrics = evaluate_model(model, _dummy_X(5), y_test)
    cm = metrics["confusion_matrix"]

    assert cm["true_negative"] == 1
    assert cm["false_positive"] == 1
    assert cm["false_negative"] == 1
    assert cm["true_positive"] == 2

    # precision = TP / (TP + FP) = 2 / 3
    assert metrics["precision"] == pytest.approx(2 / 3)
    # recall = TP / (TP + FN) = 2 / 3
    assert metrics["recall"] == pytest.approx(2 / 3)
    # accuracy = (TP + TN) / total = 3 / 5
    assert metrics["accuracy"] == pytest.approx(3 / 5)


def test_evaluate_model_confusion_matrix_values_are_native_int():
    y_test = pd.Series([0, 1])
    model = FakeModel(predictions=[0, 1], probabilities=[0.1, 0.9])

    metrics = evaluate_model(model, _dummy_X(2), y_test)
    cm = metrics["confusion_matrix"]

    # Must be plain Python int, not numpy.int64, or json.dumps() would fail
    for key in ["true_negative", "false_positive", "false_negative", "true_positive"]:
        assert isinstance(cm[key], int)
        assert not isinstance(cm[key], np.integer)


def test_evaluate_model_returns_classification_report_as_dict():
    y_test = pd.Series([0, 0, 1, 1])
    model = FakeModel(predictions=[0, 0, 1, 1], probabilities=[0.1, 0.2, 0.9, 0.95])

    metrics = evaluate_model(model, _dummy_X(4), y_test)

    assert isinstance(metrics["classification_report"], dict)
    assert "Legit" in metrics["classification_report"]
    assert "Fraud" in metrics["classification_report"]


def test_evaluate_model_includes_pr_auc_and_roc_auc():
    y_test = pd.Series([0, 0, 1, 1])
    model = FakeModel(predictions=[0, 0, 1, 1], probabilities=[0.1, 0.2, 0.9, 0.95])

    metrics = evaluate_model(model, _dummy_X(4), y_test)

    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0

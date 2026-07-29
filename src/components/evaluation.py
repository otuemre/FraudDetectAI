from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.logger import get_error_logger, log_exception

error_logger = get_error_logger()


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluates the given model on the test data and returns a dictionary of evaluation metrics.

    Args:
        model: Trained model to evaluate.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels for the test set.
    """

    try:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "pr_auc": average_precision_score(y_test, y_proba),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
            "classification_report": classification_report(
                y_test, y_pred, target_names=["Legit", "Fraud"], output_dict=True
            ),
        }

        return metrics

    except Exception as e:
        raise log_exception(error_logger, e) from e

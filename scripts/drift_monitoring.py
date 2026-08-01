import numpy as np
import pandas as pd
from sqlalchemy import text

from src.db.get_engine import get_engine
from src.logger import get_error_logger, log_exception
from src.utils import get_data

error_logger = get_error_logger()

PSI_BINS = 10
DRIFT_WATCH_THRESHOLD = 0.1
DRIFT_ALERT_THRESHOLD = 0.25


def _calculate_psi(
    expected: pd.Series, actual: pd.Series, bins: int = PSI_BINS
) -> float:
    """
    Computes PSI for a single feature. Bin edges are derived from the expected
    (training/baseline) distribution using quantiles, then reused for actual.
    """
    try:
        bin_edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        bin_edges = np.unique(
            bin_edges
        )  # guard against duplicate edges (e.g. many identical values)

        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)

        expected_pct = expected_counts / len(expected)
        actual_pct = actual_counts / len(actual)

        # Avoid division by zero / log(0) for empty bins
        epsilon = 1e-4
        expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
        actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)

    except Exception as e:
        raise log_exception(error_logger, e) from e


def get_recent_predictions(limit: int = 5000) -> pd.DataFrame:
    """Pulls the most recent N predictions from prediction_logs, unpacking input_features JSONB into columns."""
    try:
        query = text(f"""
            SELECT input_features
            FROM prediction_logs
            ORDER BY predicted_at DESC
            LIMIT {limit};
        """)

        engine = get_engine()

        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        # input_features comes back as a dict per row (JSONB) — expand into columns
        features_df = pd.json_normalize(df["input_features"])
        return features_df

    except Exception as e:
        raise log_exception(error_logger, e) from e


def run_drift_check(limit: int = 5000) -> pd.DataFrame:
    """
    Compares each feature's distribution in recent live predictions against
    the original training set, returning a PSI score and status per feature.
    """
    try:
        train_df = get_data("train_transactions")
        recent_df = get_recent_predictions(limit=limit)

        feature_cols = [
            col
            for col in recent_df.columns
            if col in train_df.columns and col != "class"
        ]

        results = []
        for col in feature_cols:
            psi = _calculate_psi(train_df[col], recent_df[col])

            if psi < DRIFT_WATCH_THRESHOLD:
                status = "stable"
            elif psi < DRIFT_ALERT_THRESHOLD:
                status = "watch"
            else:
                status = "alert"

            results.append({"feature": col, "psi": round(psi, 4), "status": status})

        results_df = (
            pd.DataFrame(results)
            .sort_values("psi", ascending=False)
            .reset_index(drop=True)
        )
        return results_df

    except Exception as e:
        raise log_exception(error_logger, e) from e


if __name__ == "__main__":
    drift_report = run_drift_check(limit=5000)
    print("Drift Report (sorted by PSI, highest first):\n")
    print(drift_report.to_string(index=False))

    alerts = drift_report[drift_report["status"] == "alert"]
    if len(alerts) > 0:
        print(
            f"\n⚠ {len(alerts)} feature(s) showing significant drift: {alerts['feature'].tolist()}"
        )
    else:
        print("\nNo significant drift detected.")

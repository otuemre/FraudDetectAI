import time

import pandas as pd

from src.components.preprocessing import raw_time_to_hours, use_scaler
from src.logger import get_error_logger, log_exception
from src.utils import load_model, load_scaler, save_prediction_log

error_logger = get_error_logger()

MODEL_VERSION = "xgboost_v1"


def predict_single(transaction: dict) -> dict:
    """
    Runs one transaction through the full prediction pipeline:
    raw input -> feature engineering -> scaling -> model prediction -> logged to SQL.
    """
    try:
        start = time.perf_counter()

        df = pd.DataFrame([transaction])

        df = raw_time_to_hours(df)

        scaler = load_scaler()
        df_scaled = use_scaler(df, scaler)

        model = load_model()
        predicted_class = int(model.predict(df_scaled)[0])
        fraud_probability = float(model.predict_proba(df_scaled)[:, 1][0])

        latency_ms = (time.perf_counter() - start) * 1000

        save_prediction_log(
            input_features=transaction,
            predicted_class=predicted_class,
            fraud_probability=fraud_probability,
            model_version=MODEL_VERSION,
            latency_ms=latency_ms,
        )

        return {
            "predicted_class": predicted_class,
            "fraud_probability": fraud_probability,
            "latency_ms": latency_ms,
        }

    except Exception as e:
        raise log_exception(error_logger, e) from e

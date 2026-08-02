import pandas as pd
from sqlalchemy import text

from src.config import CONFIG
from src.db.get_engine import get_engine
from src.logger import get_error_logger, log_exception

error_logger = get_error_logger()

LOW_CONFIDENCE_LOWER = CONFIG["monitoring"]["low_confidence_lower"]
LOW_CONFIDENCE_UPPER = CONFIG["monitoring"]["low_confidence_upper"]


def get_low_confidence_predictions(
    lower: float = LOW_CONFIDENCE_LOWER, upper: float = LOW_CONFIDENCE_UPPER
) -> pd.DataFrame:
    """
    Returns all logged predictions where fraud_probability falls within the
    'uncertain' band — i.e., the model was close to a coin flip between classes.
    """
    try:
        query = text("""
            SELECT id, predicted_at, input_features, predicted_class, fraud_probability, model_version, latency_ms
            FROM prediction_logs
            WHERE fraud_probability BETWEEN :lower AND :upper
            ORDER BY predicted_at DESC;
        """)

        engine = get_engine()

        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"lower": lower, "upper": upper})

        return df

    except Exception as e:
        raise log_exception(error_logger, e) from e


def get_daily_summary() -> pd.DataFrame:
    """
    Aggregates predictions by day: volume, average fraud probability,
    count flagged as fraud, and count landing in the low-confidence band.
    """
    try:
        query = text(f"""
            SELECT
                DATE(predicted_at) AS prediction_date,
                COUNT(*) AS total_predictions,
                SUM(predicted_class) AS flagged_as_fraud,
                ROUND(AVG(fraud_probability)::numeric, 4) AS avg_fraud_probability,
                SUM(
                    CASE WHEN fraud_probability BETWEEN {LOW_CONFIDENCE_LOWER} AND {LOW_CONFIDENCE_UPPER}
                    THEN 1 ELSE 0 END
                ) AS low_confidence_count
            FROM prediction_logs
            GROUP BY DATE(predicted_at)
            ORDER BY prediction_date DESC;
        """)

        engine = get_engine()

        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        return df

    except Exception as e:
        raise log_exception(error_logger, e) from e


if __name__ == "__main__":
    daily = get_daily_summary()
    print("Daily Summary:\n")
    print(daily.to_string(index=False))

    low_conf = get_low_confidence_predictions()
    print(
        f"\nLow-confidence predictions (band {LOW_CONFIDENCE_LOWER}-{LOW_CONFIDENCE_UPPER}): {len(low_conf)}"
    )
    print(
        low_conf[["id", "predicted_at", "fraud_probability"]]
        .head(10)
        .to_string(index=False)
    )

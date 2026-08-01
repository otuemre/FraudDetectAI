import pandas as pd
from sqlalchemy import text

from src.db.get_engine import get_engine
from src.logger import get_error_logger, log_exception

error_logger = get_error_logger()


def get_latency_summary(window: str = "day") -> pd.DataFrame:
    """
    Aggregates prediction latency by time window, reporting p50/p95/p99 and max.
    `window` controls truncation granularity: 'day' or 'hour'.
    """
    try:
        if window not in ("day", "hour"):
            raise ValueError("window must be 'day' or 'hour'")

        query = text(f"""
            SELECT
                DATE_TRUNC('{window}', predicted_at) AS time_bucket,
                COUNT(*) AS total_predictions,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms)::numeric, 2) AS p50_latency_ms,
                ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms)::numeric, 2) AS p95_latency_ms,
                ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms)::numeric, 2) AS p99_latency_ms,
                ROUND(MAX(latency_ms)::numeric, 2) AS max_latency_ms
            FROM prediction_logs
            GROUP BY time_bucket
            ORDER BY time_bucket DESC;
        """)

        engine = get_engine()

        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        return df

    except Exception as e:
        raise log_exception(error_logger, e) from e


if __name__ == "__main__":
    # Example usage
    print(get_latency_summary(window="day"))
    print(get_latency_summary(window="hour"))

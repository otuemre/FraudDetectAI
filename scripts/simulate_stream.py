import time

import pandas as pd
import requests

from src.config import CONFIG
from src.logger import get_error_logger, log_exception
from src.utils import get_data

error_logger = get_error_logger()

API_URL = "http://localhost:8000/predict"


def simulate_stream(
    limit: int | None = None, delay_seconds: float = 0.0
) -> pd.DataFrame:
    """
    Reads testing_data, sends each row to the running API's /predict endpoint one at a time,
    and returns a DataFrame combining predictions with true labels — kept locally for
    monitoring/evaluation purposes only, since a real production system wouldn't have ground truth.
    """
    try:
        test_df = get_data("test_transactions")

        if limit and limit < len(test_df):
            test_df = test_df.sample(n=limit, random_state=42)

        records = []
        total = len(test_df)

        for i, (_, row) in enumerate(test_df.iterrows(), start=1):
            true_class = int(row["class"])
            transaction = row.drop("class").to_dict()
            transaction = {
                k: float(v) for k, v in transaction.items()
            }  # numpy -> native types for JSON

            response = requests.post(API_URL, json=transaction)
            response.raise_for_status()
            result = response.json()

            records.append(
                {
                    "true_class": true_class,
                    "predicted_class": result["predicted_class"],
                    "fraud_probability": result["fraud_probability"],
                    "latency_ms": result["latency_ms"],
                }
            )

            if i % 1000 == 0 or i == total:
                print(f"Processed {i}/{total} transactions...")

            if delay_seconds:
                time.sleep(delay_seconds)

        return pd.DataFrame(records)

    except Exception as e:
        raise log_exception(error_logger, e) from e


if __name__ == "__main__":
    results = simulate_stream(
        limit=CONFIG["simulation"]["limit"],
        delay_seconds=CONFIG["simulation"]["delay_seconds"],
    )

    # Keep it if you want to learn the accuracy, however in real-world application we don't know the true label.
    accuracy = (results["true_class"] == results["predicted_class"]).mean()
    print(f"\nSimulated {len(results)} transactions — accuracy: {accuracy:.4f}")

    # Keep it if you want to save the predictions locally
    # results.to_csv("simulation_results.csv", index=False)
    # print("Saved local results to simulation_results.csv")

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import RobustScaler
from sqlalchemy import create_engine, text
from xgboost import XGBClassifier

from src.logger import get_error_logger, log_exception

error_logger = get_error_logger()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Database Connection
load_dotenv()

DB_URL = (
    f"postgresql://"
    f"{os.getenv('POSTGRES_USER', 'postgres')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'postgres')}@"
    f"localhost:5432/"
    f"{os.getenv('POSTGRES_DB', 'frauddetect')}"
)

engine = create_engine(DB_URL)


# Data Fetching and Saving Functions
def get_data(table_name: str = "transactions") -> pd.DataFrame:
    """Fetch data from the specified table in the database."""
    try:
        with engine.connect() as connection:
            query = text(f"SELECT * FROM {table_name}")
            df = pd.read_sql(query, connection)
        return df
    except Exception as e:
        raise log_exception(error_logger, e) from e


def save_data(df: pd.DataFrame, table_name: str) -> None:
    """Save DataFrame to the specified table in the database."""
    try:
        with engine.connect() as connection:
            df.to_sql(table_name, connection, if_exists="replace", index=False)
    except Exception as e:
        raise log_exception(error_logger, e) from e


# Model Saving and Loading Functions
def save_model(model: XGBClassifier, file_name: str = "final_xgb_model.json") -> None:
    """Save the trained model to a file."""
    try:
        model_path = MODELS_DIR / file_name
        model.save_model(model_path)
    except Exception as e:
        raise log_exception(error_logger, e) from e


def load_model(file_name: str = "final_xgb_model.json") -> XGBClassifier:
    """Load the trained model from a file."""
    try:
        model_path = MODELS_DIR / file_name
        model = XGBClassifier()
        model.load_model(model_path)
        return model
    except Exception as e:
        raise log_exception(error_logger, e) from e


# Scaler Saving and Loading Functions
def save_scaler(scaler: RobustScaler, file_name: str = "scaler.joblib") -> None:
    """Save the scaler to a file."""
    try:
        scaler_path = MODELS_DIR / file_name
        joblib.dump(scaler, scaler_path)
    except Exception as e:
        raise log_exception(error_logger, e) from e


def load_scaler(file_name: str = "scaler.joblib") -> RobustScaler:
    """Load the scaler from a file."""
    try:
        scaler_path = MODELS_DIR / file_name
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        raise log_exception(error_logger, e) from e


# Prediction Log
def save_prediction_log(
    input_features: dict,
    predicted_class: int,
    fraud_probability: float,
    model_version: str,
    latency_ms: float,
) -> None:
    """Inserts one prediction record into the prediction_logs table."""
    try:
        query = text("""
            INSERT INTO prediction_logs
                (input_features, predicted_class, fraud_probability, model_version, latency_ms)
            VALUES
                (:input_features, :predicted_class, :fraud_probability, :model_version, :latency_ms)
        """)

        with engine.connect() as conn:
            conn.execute(
                query,
                {
                    "input_features": json.dumps(input_features),
                    "predicted_class": predicted_class,
                    "fraud_probability": fraud_probability,
                    "model_version": model_version,
                    "latency_ms": latency_ms,
                },
            )
            conn.commit()

    except Exception as e:
        raise log_exception(error_logger, e) from e

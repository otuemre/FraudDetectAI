import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables from .env file
load_dotenv()

# Construct the database URL
DB_URL = (
    f"postgresql://"
    f"{os.getenv('POSTGRES_USER', 'postgres')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'postgres')}@"
    f"localhost:5432/"
    f"{os.getenv('POSTGRES_DB', 'frauddetect')}"
)


def create_prediction_logs_table() -> None:
    """Create the prediction_logs table if it does not already exist."""

    # Create a SQLAlchemy engine
    engine = create_engine(DB_URL)

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS prediction_logs (
        id SERIAL PRIMARY KEY,
        predicted_at TIMESTAMP NOT NULL DEFAULT NOW(),
        input_features JSONB NOT NULL,
        predicted_class INTEGER NOT NULL,
        fraud_probability FLOAT NOT NULL,
        model_version TEXT,
        latency_ms FLOAT
    );
    """

    # Execute the SQL statement
    with engine.begin() as conn:
        conn.execute(text(create_table_sql))

    print("prediction_logs table has been created.")


if __name__ == "__main__":
    create_prediction_logs_table()

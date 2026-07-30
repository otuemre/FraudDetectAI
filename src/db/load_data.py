import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load environment variables from .env file
load_dotenv()

# Construct the database URL using environment variables
DB_URL = (
    f"postgresql://"
    f"{os.getenv('POSTGRES_USER', 'postgres')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'postgres')}@"
    f"localhost:5432/"
    f"{os.getenv('POSTGRES_DB', 'frauddetect')}"
)


def load_csv_to_db(csv_path: str, table_name: str) -> None:
    """Load a CSV file into a PostgreSQL table."""

    # Create a SQLAlchemy engine
    engine = create_engine(DB_URL)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Normalize feature names (Convert to lowercase)
    df.columns = df.columns.str.lower()

    # Load the DataFrame into the database
    df.to_sql(table_name, engine, if_exists="replace", index=False, chunksize=10000)

    print(f"Data from {csv_path} has been loaded into the {table_name} table.")


if __name__ == "__main__":
    # Example usage: Load data from a CSV file into the 'transactions' table
    load_csv_to_db("src/datasets/creditcard.csv", "transactions")

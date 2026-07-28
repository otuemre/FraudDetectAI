import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from src.logger import get_error_logger, log_exception

error_logger = get_error_logger()

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

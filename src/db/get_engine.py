import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

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


def get_engine():
    """
    Get the enginer
    """
    try:
        engine = create_engine(DB_URL)
        return engine
    except Exception as e:
        raise log_exception(error_logger, e) from e

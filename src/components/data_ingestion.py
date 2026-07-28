from sklearn.model_selection import train_test_split

from src.logger import get_error_logger, log_exception
from src.utils import get_data, save_data

error_logger = get_error_logger()

SEED = 42
TEST_SIZE = 0.2


def ingest_data(source_table: str = "transactions") -> None:
    """
    Fetches data from the specified source table, splits it into training and testing sets,
    and saves them back to the database.
    """
    try:
        # Fetch data
        df = get_data(source_table)

        # Drop Duplicates
        df = df.drop_duplicates().reset_index(drop=True)

        # Split data into training and testing sets
        train_df, test_df = train_test_split(
            df, test_size=TEST_SIZE, random_state=SEED, stratify=df["class"]
        )

        # Save the split datasets back to the database
        save_data(train_df, "train_transactions")
        save_data(test_df, "test_transactions")

    except Exception as e:
        raise log_exception(error_logger, e) from e

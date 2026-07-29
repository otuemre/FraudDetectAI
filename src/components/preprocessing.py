import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.logger import get_error_logger, log_exception

error_logger = get_error_logger()

SCALE_COLS = ["amount", "hour_of_day"]


# Feature Engineering
def raw_time_to_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the 'time' column in the DataFrame to 'hour_of_day'.
    """
    try:
        df = df.copy()
        df["hour_of_day"] = (df["time"] % 86400) / 3600
        df = df.drop(columns=["time"])
        return df
    except Exception as e:
        raise log_exception(error_logger, e) from e


# Scaler
def train_scaler(df: pd.DataFrame) -> RobustScaler:
    """
    Trains a RobustScaler on the specified columns of the DataFrame.
    """
    try:
        scaler = RobustScaler()
        scaler.fit(df[SCALE_COLS])
        return scaler
    except Exception as e:
        raise log_exception(error_logger, e) from e


def use_scaler(df: pd.DataFrame, scaler: RobustScaler) -> pd.DataFrame:
    """
    Applies the provided scaler to the specified columns of the DataFrame.
    """
    try:
        df = df.copy()
        df[SCALE_COLS] = scaler.transform(df[SCALE_COLS])
        return df
    except Exception as e:
        raise log_exception(error_logger, e) from e

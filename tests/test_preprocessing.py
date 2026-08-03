import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from src.components.preprocessing import raw_time_to_hours, train_scaler, use_scaler


def _sample_df() -> pd.DataFrame:
    """A tiny DataFrame with 'time' and 'amount', enough to exercise both functions."""
    return pd.DataFrame(
        {
            "time": [
                0,
                3600,
                90000,
                172800,
            ],  # 0h, 1h, 25h -> wraps to 1h, 48h -> wraps to 0h
            "amount": [10.0, 500.0, 25.0, 1000.0],
            "v1": [0.1, -0.2, 0.3, -0.4],
        }
    )


def test_raw_time_to_hours_computes_correct_values():
    df = _sample_df()
    result = raw_time_to_hours(df)

    assert "hour_of_day" in result.columns
    assert "time" not in result.columns

    expected_hours = [
        0.0,
        1.0,
        1.0,
        0.0,
    ]
    assert result["hour_of_day"].tolist() == pytest.approx(expected_hours)


def test_raw_time_to_hours_does_not_mutate_original():
    df = _sample_df()
    original_columns = df.columns.tolist()

    raw_time_to_hours(df)

    # original df passed in should be untouched
    assert df.columns.tolist() == original_columns
    assert "time" in df.columns


def test_hour_of_day_within_valid_range():
    df = _sample_df()
    result = raw_time_to_hours(df)

    assert (result["hour_of_day"] >= 0).all()
    assert (result["hour_of_day"] < 24).all()


def test_train_scaler_returns_fitted_robust_scaler():
    df = _sample_df()
    df = raw_time_to_hours(df)

    scaler = train_scaler(df)

    assert isinstance(scaler, RobustScaler)
    # A fitted scaler has center_/scale_ attributes populated
    assert scaler.center_ is not None
    assert len(scaler.center_) == 2  # amount, hour_of_day


def test_use_scaler_transforms_expected_columns_only():
    df = _sample_df()
    df = raw_time_to_hours(df)

    scaler = train_scaler(df)
    scaled_df = use_scaler(df, scaler)

    # v1 (not a scale column) should be untouched
    assert scaled_df["v1"].tolist() == df["v1"].tolist()

    # amount/hour_of_day should have changed from their raw values
    assert not scaled_df["amount"].equals(df["amount"])


def test_use_scaler_does_not_mutate_input():
    df = _sample_df()
    df = raw_time_to_hours(df)
    scaler = train_scaler(df)

    original_amount = df["amount"].copy()
    use_scaler(df, scaler)

    assert df["amount"].equals(original_amount)


def test_use_scaler_is_deterministic():
    df = _sample_df()
    df = raw_time_to_hours(df)
    scaler = train_scaler(df)

    result_1 = use_scaler(df, scaler)
    result_2 = use_scaler(df, scaler)

    pd.testing.assert_frame_equal(result_1, result_2)

from xgboost import XGBClassifier

from src.components.data_ingestion import ingest_data
from src.components.evaluation import evaluate_model
from src.components.hyperparameter_tuning import tune_hyperparameters
from src.components.preprocessing import raw_time_to_hours, train_scaler, use_scaler
from src.config import CONFIG
from src.logger import get_error_logger, get_training_logger, log_exception
from src.utils import get_data, save_model, save_scaler

error_logger = get_error_logger()


def training_pipeline(n_trials: int = 5) -> dict:
    """
    Trains a model using the specified number of trials.

    Args:
        n_trials (int, optional): _description_. Defaults to 5.

    Returns:
        dict: _description_
    """

    training_logger = get_training_logger()

    try:
        training_logger.info({"message": "Training pipeline started"})

        # Ingest data
        ingest_data(source_table="transactions")
        training_logger.info({"message": "Data ingestion completed"})

        train_df = get_data(table_name="train_transactions")
        test_df = get_data(table_name="test_transactions")
        training_logger.info(
            {
                "message": "Loaded train/test tables",
                "train_shape": list(train_df.shape),
                "test_shape": list(test_df.shape),
            }
        )

        # Feature engineering
        train_df = raw_time_to_hours(train_df)
        test_df = raw_time_to_hours(test_df)

        X_train, y_train = train_df.drop(columns=["class"]), train_df["class"]
        X_test, y_test = test_df.drop(columns=["class"]), test_df["class"]
        training_logger.info({"message": "Feature engineering complete"})

        # Scaling
        scaler = train_scaler(X_train)
        X_train = use_scaler(X_train, scaler)
        X_test = use_scaler(X_test, scaler)
        training_logger.info({"message": "Preprocessing complete"})

        # Hyperparameter Tuning
        best_params = tune_hyperparameters(X_train, y_train, n_trials=n_trials)
        training_logger.info(
            {"message": "Hyperparameter tuning complete", "best_params": best_params}
        )

        # Model Fit
        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        training_logger.info({"message": "Final model fit complete"})

        # Evaluation
        metrics = evaluate_model(model, X_test, y_test)
        training_logger.info({"message": "Evaluation complete", **metrics})

        # Save Artifacts
        save_model(model)
        save_scaler(scaler)
        training_logger.info({"message": "Model and scaler saved"})

        training_logger.info({"message": "Training pipeline finished successfully"})
        return metrics

    except Exception as e:
        raise log_exception(error_logger, e) from e


if __name__ == "__main__":
    N_TRIALS = CONFIG["training"]["n_trials"]
    results = training_pipeline(n_trials=N_TRIALS)
    print(results)

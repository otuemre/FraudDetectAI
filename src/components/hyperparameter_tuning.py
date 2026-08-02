import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src.config import CONFIG
from src.logger import get_error_logger, log_exception

error_logger = get_error_logger()

SEED = CONFIG["training"]["seed"]
N_SPLITS = CONFIG["training"]["n_splits"]
N_TRIALS = CONFIG["training"]["n_trials"]


def _build_objective(X_train, y_train, cv, scale_pos_weight_base):
    """Returns an Optuna objective closure bound to the given training data and CV strategy."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight",
                scale_pos_weight_base * 0.5,
                scale_pos_weight_base * 1.5,
            ),
            "random_state": SEED,
            "eval_metric": "aucpr",
        }

        model = XGBClassifier(**params)

        scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1
        )

        return scores.mean()

    return objective


def tune_hyperparameters(X_train, y_train, n_trials: int = N_TRIALS) -> dict:
    """
    Tunes hyperparameters for an XGBoost model using Optuna.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        n_trials (int): Number of trials for hyperparameter tuning.

    Returns:
        dict: Best hyperparameters found during tuning.
    """
    try:
        # Calculate scale_pos_weight based on class imbalance
        scale_pos_weight_base = (y_train == 0).sum() / (y_train == 1).sum()

        # Define cross-validation strategy
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

        # Build the objective
        objective = _build_objective(X_train, y_train, cv, scale_pos_weight_base)

        # Set Optuna logging level to WARNING to reduce verbosity
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Create Optuna study
        sampler = TPESampler(seed=SEED)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name="xgb_hyperparameter_tuning",
        )

        # Optimize the objective function
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
        )

        best_params = study.best_params
        best_params.update(
            {
                "random_state": SEED,
                "eval_metric": "aucpr",
            }
        )

        return best_params

    except Exception as e:
        raise log_exception(error_logger, e) from e

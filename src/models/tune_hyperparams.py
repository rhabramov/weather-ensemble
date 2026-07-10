"""
Hyperparameter tuning for the weather ensemble XGBoost models.

Uses Optuna for Bayesian optimization with time-series cross-validation.
Typically finds 0.1-0.2F improvement over default params.

Install: pip install optuna
Run: python src/models/tune_hyperparams.py data/processed/training_data.csv
"""

import logging
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

from config.settings import MODELS_DIR
from src.models.xgb_model import prepare_training_data, get_feature_cols

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("Install optuna first: pip install optuna")
    sys.exit(1)


def objective(trial, X, y, n_splits=5):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 400, 1500),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth":         trial.suggest_int("max_depth", 4, 9),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 5.0),
        "gamma":             trial.suggest_float("gamma", 0.0, 1.0),
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []
    for tr_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        maes.append(mean_absolute_error(y_val, model.predict(X_val)))

    return np.mean(maes)


def tune_model(df: pd.DataFrame, target: str, n_trials: int = 100):
    logger.info(f"\n{'='*50}")
    logger.info(f"Tuning {target} model with {n_trials} trials...")

    df = df.sort_values("forecast_date").reset_index(drop=True)
    X, y, city_encoder = prepare_training_data(df, target)

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    best_mae = study.best_value
    logger.info(f"Best CV MAE: {best_mae:.4f}°F")
    logger.info(f"Best params: {best}")

    # Retrain final model on all data with best params
    final_params = {**best, "random_state": 42, "n_jobs": -1, "tree_method": "hist"}
    final_model = xgb.XGBRegressor(**final_params)
    final_model.fit(X, y, verbose=False)

    # Save
    suffix = "high" if "high" in target else "low"
    model_path = MODELS_DIR / f"model_{suffix}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    logger.info(f"Saved tuned model → {model_path}")

    # Save best params for reference
    params_path = MODELS_DIR / f"best_params_{suffix}.json"
    import json
    with open(params_path, "w") as f:
        json.dump(best, f, indent=2)

    return final_model, city_encoder, best_mae


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/models/tune_hyperparams.py data/processed/training_data.csv")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1], parse_dates=["forecast_date"])
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    results = {}
    for target in ["actual_high", "actual_low"]:
        model, encoder, mae = tune_model(df, target, n_trials)
        results[target] = mae

    # Save encoder (shared)
    from src.models.xgb_model import prepare_training_data, ENCODER_PATH
    _, _, city_encoder = prepare_training_data(df.sort_values("forecast_date"), "actual_high")
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(city_encoder, f)

    print(f"\n=== Tuning Results ===")
    print(f"High CV MAE: {results['actual_high']:.4f}°F")
    print(f"Low  CV MAE: {results['actual_low']:.4f}°F")
    print(f"\nModels saved to {MODELS_DIR}")
    print("Commit the new .pkl files and push to GitHub.")
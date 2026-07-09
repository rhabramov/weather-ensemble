"""
XGBoost ensemble model — training and inference.

Two models trained:
  - model_high: predicts daily high temperature
  - model_low:  predicts daily low temperature

Training data: historical feature matrices joined to CLI-verified labels.
The model takes the 34 forecast members + ensemble stats + temporal/city
features and learns to predict the NWS CLI verified high/low.

Approach: global model with city as a categorical feature. City-level
residual analysis is logged so systematic bias can be detected.
"""

import logging
import pickle
from datetime import date
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from config.settings import MODELS_DIR

logger = logging.getLogger(__name__)

MODEL_HIGH_PATH = MODELS_DIR / "model_high.pkl"
MODEL_LOW_PATH  = MODELS_DIR / "model_low.pkl"
ENCODER_PATH    = MODELS_DIR / "city_encoder.pkl"

# Features to exclude from the model (metadata + labels)
EXCLUDE_COLS = {
    "city", "forecast_date", "date",
    "actual_high", "actual_low",
}

XGB_PARAMS = {
    "n_estimators":     800,
    "learning_rate":    0.05,
    "max_depth":        6,
    "min_child_weight": 3,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "n_jobs":           -1,
    "tree_method":      "hist",
}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def prepare_training_data(
    df: pd.DataFrame,
    label_col: str,
    city_encoder: Optional[LabelEncoder] = None,
) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """
    Prepare X, y for training.

    - Encode city as integer
    - Drop rows where label is missing
    - Fill remaining NaN features with column median (missing members)
    """
    df = df.copy()
    df = df.dropna(subset=[label_col])

    # City encoding
    if city_encoder is None:
        city_encoder = LabelEncoder()
        df["city_idx"] = city_encoder.fit_transform(df["city"])
    else:
        df["city_idx"] = city_encoder.transform(df["city"])

    feature_cols = get_feature_cols(df)
    X = df[feature_cols].copy()

    # Fill missing forecast members with median of that column
    # (happens when an API is down — this makes the model robust)
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    y = df[label_col]
    return X, y, city_encoder


def train_models(
    df: pd.DataFrame,
    test_fraction: float = 0.15,
    save: bool = True,
) -> dict:
    """
    Train high and low models with time-series cross-validation.

    df must contain: all feature columns + 'actual_high' + 'actual_low'
    sorted chronologically.

    Returns dict with models, encoders, and evaluation metrics.
    """
    logger.info(f"Training on {len(df)} samples across {df['city'].nunique()} cities")

    # Time-based split — never random
    df = df.sort_values("forecast_date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_fraction))
    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]

    results = {}

    for target, model_path in [("actual_high", MODEL_HIGH_PATH), ("actual_low", MODEL_LOW_PATH)]:
        logger.info(f"\n--- Training {target} model ---")

        X_train, y_train, city_encoder = prepare_training_data(train_df, target)
        X_test,  y_test,  _            = prepare_training_data(test_df, target, city_encoder)

        model = xgb.XGBRegressor(**XGB_PARAMS)

        # Time-series CV on training set for early stopping reference
        tscv = TimeSeriesSplit(n_splits=5)
        cv_maes = []
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            m = xgb.XGBRegressor(**XGB_PARAMS)
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            pred = m.predict(X_val)
            mae = mean_absolute_error(y_val, pred)
            cv_maes.append(mae)
            logger.info(f"  CV fold {fold+1}: MAE = {mae:.3f}°F")

        logger.info(f"  CV mean MAE: {np.mean(cv_maes):.3f}°F ± {np.std(cv_maes):.3f}")

        # Final fit on full training set
        model.fit(X_train, y_train, verbose=False)

        # Evaluate on held-out test set
        test_preds = model.predict(X_test)
        test_mae   = mean_absolute_error(y_test, test_preds)
        test_rmse  = np.sqrt(mean_squared_error(y_test, test_preds))
        test_bias  = float(np.mean(test_preds - y_test))

        logger.info(f"  Test MAE:  {test_mae:.3f}°F")
        logger.info(f"  Test RMSE: {test_rmse:.3f}°F")
        logger.info(f"  Test Bias: {test_bias:.3f}°F")

        # Per-city breakdown
        test_df_copy = test_df.copy()
        test_df_copy[f"pred_{target}"] = test_preds
        for city in test_df_copy["city"].unique():
            city_mask = test_df_copy["city"] == city
            city_mae  = mean_absolute_error(
                test_df_copy.loc[city_mask, target],
                test_df_copy.loc[city_mask, f"pred_{target}"]
            )
            city_bias = float(np.mean(
                test_df_copy.loc[city_mask, f"pred_{target}"]
                - test_df_copy.loc[city_mask, target]
            ))
            logger.info(f"    {city:20s}  MAE={city_mae:.2f}°F  Bias={city_bias:+.2f}°F")

        # Feature importance (top 15)
        feat_imp = pd.Series(
            model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)
        logger.info(f"\n  Top 15 features for {target}:")
        for feat, imp in feat_imp.head(15).items():
            logger.info(f"    {feat:40s}  {imp:.4f}")

        if save:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"  Saved model to {model_path}")

        results[target] = {
            "model": model,
            "city_encoder": city_encoder,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_bias": test_bias,
            "cv_mae_mean": float(np.mean(cv_maes)),
            "feature_importance": feat_imp.to_dict(),
        }

    if save:
        with open(ENCODER_PATH, "wb") as f:
            pickle.dump(results["actual_high"]["city_encoder"], f)

    return results


def load_models() -> Tuple[xgb.XGBRegressor, xgb.XGBRegressor, LabelEncoder]:
    """Load trained high and low models from disk."""
    with open(MODEL_HIGH_PATH, "rb") as f:
        model_high = pickle.load(f)
    with open(MODEL_LOW_PATH, "rb") as f:
        model_low = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        city_encoder = pickle.load(f)
    return model_high, model_low, city_encoder


def predict(
    feature_df: pd.DataFrame,
    model_high: Optional[xgb.XGBRegressor] = None,
    model_low:  Optional[xgb.XGBRegressor] = None,
    city_encoder: Optional[LabelEncoder] = None,
) -> pd.DataFrame:
    """
    Run inference on a feature DataFrame (one row per city).

    Returns DataFrame with city, forecast_date, pred_high, pred_low.
    Loads models from disk if not passed in.
    """
    if model_high is None or model_low is None or city_encoder is None:
        model_high, model_low, city_encoder = load_models()

    df = feature_df.copy()
    df["city_idx"] = city_encoder.transform(df["city"])

    feature_cols = get_feature_cols(df)
    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    preds = df[["city", "forecast_date"]].copy()
    preds["pred_high"] = np.round(model_high.predict(X), 1)
    preds["pred_low"]  = np.round(model_low.predict(X), 1)
    preds["run_time"]  = pd.Timestamp.now(tz="UTC").isoformat()

    return preds


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    # Example: load a pre-built training CSV and train
    if len(sys.argv) > 1:
        df = pd.read_csv(sys.argv[1], parse_dates=["forecast_date"])
        results = train_models(df)
        print("\nTraining complete.")
        print(f"High model test MAE: {results['actual_high']['test_mae']:.3f}°F")
        print(f"Low  model test MAE: {results['actual_low']['test_mae']:.3f}°F")
    else:
        print("Usage: python xgb_model.py <training_data.csv>")

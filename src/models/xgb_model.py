"""
XGBoost ensemble model — training and inference.

Pipeline:
  1. Global XGBoost model (high + low separately)
  2. City-specific monthly bias correction layer

Loads best hyperparams from tune_hyperparams.py output if available,
otherwise uses strong defaults.
"""

import logging
import pickle
import json
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
from src.models.bias_correction import BiasCorrector

logger = logging.getLogger(__name__)

MODEL_HIGH_PATH   = MODELS_DIR / "model_high.pkl"
MODEL_LOW_PATH    = MODELS_DIR / "model_low.pkl"
ENCODER_PATH      = MODELS_DIR / "city_encoder.pkl"
BEST_PARAMS_HIGH  = MODELS_DIR / "best_params_high.json"
BEST_PARAMS_LOW   = MODELS_DIR / "best_params_low.json"

EXCLUDE_COLS = {
    "city", "forecast_date", "date",
    "actual_high", "actual_low",
    "era5_high", "era5_low",   # proxy features — excluded to reduce leakage signal
}

DEFAULT_PARAMS = {
    "n_estimators":     800,
    "learning_rate":    0.05,
    "max_depth":        6,
    "min_child_weight": 3,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "gamma":            0.0,
    "random_state":     42,
    "n_jobs":           -1,
    "tree_method":      "hist",
}


def load_best_params(target: str) -> dict:
    """Load tuned hyperparams from JSON if available, else use defaults."""
    suffix     = "high" if "high" in target else "low"
    path       = BEST_PARAMS_HIGH if suffix == "high" else BEST_PARAMS_LOW
    if path.exists():
        with open(path) as f:
            params = json.load(f)
        logger.info(f"Loaded tuned params for {target} from {path}")
        params.update({"random_state": 42, "n_jobs": -1, "tree_method": "hist"})
        return params
    logger.info(f"No tuned params found for {target} — using defaults")
    return DEFAULT_PARAMS.copy()


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def prepare_training_data(
    df: pd.DataFrame,
    label_col: str,
    city_encoder: Optional[LabelEncoder] = None,
) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    df = df.copy()
    df = df.dropna(subset=[label_col])

    if city_encoder is None:
        city_encoder = LabelEncoder()
        df["city_idx"] = city_encoder.fit_transform(df["city"])
    else:
        df["city_idx"] = city_encoder.transform(df["city"])

    feature_cols = get_feature_cols(df)
    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    return X, df[label_col], city_encoder


def train_models(df: pd.DataFrame, test_fraction: float = 0.15, save: bool = True) -> dict:
    """Train high and low models with time-series CV and bias correction."""
    logger.info(f"Training on {len(df)} samples | {df['city'].nunique()} cities")

    df = df.sort_values("forecast_date").reset_index(drop=True)
    split_idx  = int(len(df) * (1 - test_fraction))
    train_df   = df.iloc[:split_idx]
    test_df    = df.iloc[split_idx:]

    results    = {}
    all_test_preds = {}

    for target, model_path in [("actual_high", MODEL_HIGH_PATH),
                                ("actual_low",  MODEL_LOW_PATH)]:
        logger.info(f"\n{'='*50}\nTraining {target} model")

        params = load_best_params(target)
        X_train, y_train, city_encoder = prepare_training_data(train_df, target)
        X_test,  y_test,  _            = prepare_training_data(test_df,  target, city_encoder)

        # Time-series CV
        tscv    = TimeSeriesSplit(n_splits=5)
        cv_maes = []
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
            m = xgb.XGBRegressor(**params)
            m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx],
                  eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
                  verbose=False)
            mae = mean_absolute_error(y_train.iloc[val_idx], m.predict(X_train.iloc[val_idx]))
            cv_maes.append(mae)
            logger.info(f"  CV fold {fold+1}: MAE={mae:.3f}°F")

        logger.info(f"  CV mean MAE: {np.mean(cv_maes):.3f}°F ± {np.std(cv_maes):.3f}")

        # Final fit on full train set
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)

        test_preds = model.predict(X_test)
        test_mae   = mean_absolute_error(y_test, test_preds)
        test_rmse  = np.sqrt(mean_squared_error(y_test, test_preds))
        test_bias  = float(np.mean(test_preds - y_test))
        logger.info(f"  Test MAE={test_mae:.3f}°F  RMSE={test_rmse:.3f}°F  Bias={test_bias:+.3f}°F")

        # Per-city breakdown
        suffix = "high" if "high" in target else "low"
        test_copy = test_df.copy()
        test_copy[f"pred_{suffix}"] = test_preds
        test_copy[f"actual_{suffix}"] = y_test.values
        all_test_preds[suffix] = test_copy

        for city in sorted(test_copy["city"].unique()):
            mask     = test_copy["city"] == city
            city_mae = mean_absolute_error(
                test_copy.loc[mask, f"actual_{suffix}"],
                test_copy.loc[mask, f"pred_{suffix}"]
            )
            city_bias = float(np.mean(
                test_copy.loc[mask, f"pred_{suffix}"] -
                test_copy.loc[mask, f"actual_{suffix}"]
            ))
            logger.info(f"    {city:20s}  MAE={city_mae:.2f}°F  Bias={city_bias:+.2f}°F")

        # Top features
        feat_imp = pd.Series(
            model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)
        logger.info(f"\n  Top 10 features:")
        for feat, imp in feat_imp.head(10).items():
            logger.info(f"    {feat:40s}  {imp:.4f}")

        if save:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        results[target] = {
            "model": model, "city_encoder": city_encoder,
            "test_mae": test_mae, "cv_mae_mean": float(np.mean(cv_maes)),
        }

    if save:
        with open(ENCODER_PATH, "wb") as f:
            pickle.dump(results["actual_high"]["city_encoder"], f)

    # Fit bias corrector on test set residuals
    logger.info("\n=== Fitting bias corrector ===")
    test_high = all_test_preds["high"][["city", "forecast_date", "pred_high", "actual_high"]]
    test_low  = all_test_preds["low"][["city", "forecast_date", "pred_low",  "actual_low"]]
    bias_df   = test_high.merge(test_low, on=["city", "forecast_date"])

    corrector = BiasCorrector()
    corrector.fit(bias_df)
    metrics = corrector.evaluate(bias_df)

    if save:
        corrector.save()

    logger.info(f"\n=== Final Results ===")
    logger.info(f"High: XGB MAE={results['actual_high']['test_mae']:.3f}°F  "
                f"→ after bias correction: {metrics['mae_high_after']:.3f}°F")
    logger.info(f"Low:  XGB MAE={results['actual_low']['test_mae']:.3f}°F  "
                f"→ after bias correction: {metrics['mae_low_after']:.3f}°F")

    return results


def load_models() -> Tuple[xgb.XGBRegressor, xgb.XGBRegressor, LabelEncoder, BiasCorrector]:
    with open(MODEL_HIGH_PATH, "rb") as f:
        model_high = pickle.load(f)
    with open(MODEL_LOW_PATH, "rb") as f:
        model_low = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        city_encoder = pickle.load(f)
    corrector = BiasCorrector.load()
    return model_high, model_low, city_encoder, corrector


def predict(
    feature_df: pd.DataFrame,
    model_high: Optional[xgb.XGBRegressor] = None,
    model_low:  Optional[xgb.XGBRegressor] = None,
    city_encoder: Optional[LabelEncoder]   = None,
    corrector: Optional[BiasCorrector]     = None,
) -> pd.DataFrame:
    """Run inference and apply bias correction."""
    if model_high is None:
        model_high, model_low, city_encoder, corrector = load_models()

    df = feature_df.copy()
    df["city_idx"] = city_encoder.transform(df["city"])

    feature_cols = get_feature_cols(df)
    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    # Align feature columns to match what the model was trained on
    expected_features = model_high.get_booster().feature_names

    # Add missing columns as NaN
    for col in expected_features:
        if col not in X.columns:
            logger.warning("Missing feature column: %s — filling with NaN", col)
            X[col] = np.nan

    # Drop extra columns, reorder to match training
    X = X[expected_features]
    X = X.astype({col: "float64" for col in X.columns if X[col].dtype == object})

    preds = df[["city", "forecast_date"]].copy()
    preds["pred_high"] = np.round(model_high.predict(X), 1)
    preds["pred_low"]  = np.round(model_low.predict(X), 1)
    preds["run_time"]  = pd.Timestamp.now(tz="UTC").isoformat()

    # Apply bias correction
    if corrector is not None:
        preds = corrector.apply(preds)

    return preds


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if len(sys.argv) > 1:
        df = pd.read_csv(sys.argv[1], parse_dates=["forecast_date"])
        results = train_models(df)
    else:
        print("Usage: python src/models/xgb_model.py <training_data.csv>")
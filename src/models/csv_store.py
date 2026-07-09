"""
CSV-based persistent store.

Replaces the SQLAlchemy database layer with plain CSV files that are
committed back to the GitHub repo after each run.

Files (all under data_store/):
  predictions.csv   — every prediction ever made
  actuals.csv       — CLI-verified highs/lows (ground truth)
  run_log.csv       — one row per run for monitoring

Why CSVs over SQLite for this use case:
  - SQLite is a binary file; git diffs are unreadable
  - CSVs are human-readable in GitHub's UI
  - 2 runs/day = ~700 rows/year added to predictions.csv — trivial size
  - No external DB account needed
"""

import logging
import os
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data_store"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_CSV = DATA_DIR / "predictions.csv"
ACTUALS_CSV     = DATA_DIR / "actuals.csv"
RUN_LOG_CSV     = DATA_DIR / "run_log.csv"

PREDICTIONS_COLS = [
    "city", "forecast_date", "run_time", "run_type", "pred_high", "pred_low"
]
ACTUALS_COLS = [
    "city", "report_date", "actual_high", "actual_low", "ingested_at"
]
RUN_LOG_COLS = [
    "run_time", "run_type", "status", "n_cities", "duration_s", "notes"
]


def _load(path: Path, cols: list[str]) -> pd.DataFrame:
    """Load a CSV or return empty DataFrame with correct columns."""
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=cols)


def _save(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)


def save_predictions(preds_df: pd.DataFrame, run_type: str = "morning"):
    """Append today's predictions to predictions.csv."""
    existing = _load(PREDICTIONS_CSV, PREDICTIONS_COLS)
    now = datetime.now(timezone.utc).isoformat()

    new_rows = preds_df[["city", "forecast_date"]].copy()
    new_rows["run_time"]  = now
    new_rows["run_type"]  = run_type
    new_rows["pred_high"] = preds_df["pred_high"].values
    new_rows["pred_low"]  = preds_df["pred_low"].values

    combined = pd.concat([existing, new_rows], ignore_index=True)
    _save(combined, PREDICTIONS_CSV)
    logger.info(f"Saved {len(new_rows)} predictions to {PREDICTIONS_CSV}")


def save_actuals(cli_reports: list[dict]):
    """
    Upsert CLI actuals into actuals.csv.
    Updates existing row if city+date already present.
    """
    existing = _load(ACTUALS_CSV, ACTUALS_COLS)
    now = datetime.now(timezone.utc).isoformat()

    new_rows = []
    for r in cli_reports:
        if not r.get("report_date"):
            continue
        if r.get("actual_high") is None and r.get("actual_low") is None:
            continue
        new_rows.append({
            "city":        r["city"],
            "report_date": str(r["report_date"]),
            "actual_high": r.get("actual_high"),
            "actual_low":  r.get("actual_low"),
            "ingested_at": now,
        })

    if not new_rows:
        logger.warning("No actuals to save")
        return

    new_df = pd.DataFrame(new_rows)

    # Upsert: drop existing rows for same city+date, then append
    if not existing.empty:
        mask = existing.set_index(["city", "report_date"]).index.isin(
            new_df.set_index(["city", "report_date"]).index
        )
        existing = existing[~mask]

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.sort_values(["report_date", "city"]).reset_index(drop=True)
    _save(combined, ACTUALS_CSV)
    logger.info(f"Saved {len(new_rows)} actuals to {ACTUALS_CSV}")


def log_run(run_type: str, status: str, n_cities: int, duration_s: float, notes: str = ""):
    """Append a run record to run_log.csv."""
    existing = _load(RUN_LOG_CSV, RUN_LOG_COLS)
    new_row = pd.DataFrame([{
        "run_time":   datetime.now(timezone.utc).isoformat(),
        "run_type":   run_type,
        "status":     status,
        "n_cities":   n_cities,
        "duration_s": round(duration_s, 1),
        "notes":      notes,
    }])
    combined = pd.concat([existing, new_row], ignore_index=True)
    _save(combined, RUN_LOG_CSV)


def get_recent_predictions(days: int = 7) -> pd.DataFrame:
    """
    Return predictions joined to actuals for the last N days.
    Useful for accuracy reporting in the email.
    """
    preds   = _load(PREDICTIONS_CSV, PREDICTIONS_COLS)
    actuals = _load(ACTUALS_CSV, ACTUALS_COLS)

    if preds.empty:
        return pd.DataFrame()

    preds["forecast_date"] = pd.to_datetime(preds["forecast_date"]).dt.date
    cutoff = pd.Timestamp.now().date() - pd.Timedelta(days=days)
    preds = preds[preds["forecast_date"] >= cutoff]

    if actuals.empty:
        return preds

    actuals["report_date"] = pd.to_datetime(actuals["report_date"]).dt.date
    merged = preds.merge(
        actuals[["city", "report_date", "actual_high", "actual_low"]],
        left_on=["city", "forecast_date"],
        right_on=["city", "report_date"],
        how="left",
    )
    merged["error_high"] = merged["pred_high"] - merged["actual_high"]
    merged["error_low"]  = merged["pred_low"]  - merged["actual_low"]
    return merged


def get_yesterday_accuracy() -> Optional[pd.DataFrame]:
    """Return the morning predictions for yesterday joined to actuals, for the email."""
    df = get_recent_predictions(days=2)
    if df.empty:
        return None
    yesterday = (pd.Timestamp.now().date() - pd.Timedelta(days=1))
    mask = (df["forecast_date"] == yesterday) & (df["run_type"] == "morning")
    result = df[mask]
    return result if not result.empty else None

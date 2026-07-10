"""
Climatological normal features.

Fetches 1991-2020 30-year daily normals from Open-Meteo ERA5 climate API
for each city. Computes day-of-year averages and saves to
data/processed/climo_normals.csv.

Features added:
  climo_high      — 30yr normal high for this day of year
  climo_low       — 30yr normal low for this day of year
  anomaly_high    — era5_high - climo_high (how unusual is today vs history)
  anomaly_low     — era5_low  - climo_low

These are among the strongest supplementary predictors because the model
learns that a day 15°F above normal behaves differently than a normal day
at the same absolute temperature.

Run once:
  python src/features/climo_features.py
"""

import logging
import time
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
import numpy as np

from config.cities import CITIES
from config.settings import PROCESSED_DIR

logger = logging.getLogger(__name__)

CLIMO_URL    = "https://archive-api.open-meteo.com/v1/archive"
CLIMO_PATH   = PROCESSED_DIR / "climo_normals.csv"

# 1991-2020 is the WMO standard climate normal period
CLIMO_START  = "1991-01-01"
CLIMO_END    = "2020-12-31"


def fetch_climo_city(city_name: str, city_cfg: dict) -> pd.DataFrame:
    """
    Fetch 30 years of ERA5 daily high/low for one city and average
    by day-of-year to get climatological normals.
    """
    params = {
        "latitude":         city_cfg["lat"],
        "longitude":        city_cfg["lon"],
        "daily":            "temperature_2m_max,temperature_2m_min",
        "temperature_unit": "fahrenheit",
        "timezone":         city_cfg["tz"],
        "start_date":       CLIMO_START,
        "end_date":         CLIMO_END,
    }

    for attempt in range(1, 4):
        try:
            r = httpx.get(CLIMO_URL, params=params, timeout=60)
            if r.status_code == 429:
                wait = 30 * attempt
                logger.warning(f"429 for {city_name} — waiting {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            daily = r.json().get("daily", {})
            df = pd.DataFrame({
                "date":  pd.to_datetime(daily.get("time", [])),
                "high":  daily.get("temperature_2m_max", []),
                "low":   daily.get("temperature_2m_min", []),
            })
            df["doy"] = df["date"].dt.dayofyear
            normals = df.groupby("doy")[["high", "low"]].mean().reset_index()
            normals.columns = ["doy", "climo_high", "climo_low"]
            normals["city"] = city_name
            logger.info(f"  {city_name:20s}: {len(df)} days → {len(normals)} DOY normals")
            return normals[["city", "doy", "climo_high", "climo_low"]]
        except Exception as e:
            logger.error(f"  {city_name} attempt {attempt}: {e}")
            if attempt < 3:
                time.sleep(15)

    return pd.DataFrame()


def build_climo_normals(output_path: Optional[Path] = None) -> pd.DataFrame:
    """Fetch climo normals for all 20 cities and save to CSV."""
    if output_path is None:
        output_path = CLIMO_PATH

    logger.info(f"Fetching 1991-2020 climatological normals for {len(CITIES)} cities...")
    all_dfs = []
    items = list(CITIES.items())

    for i, (city_name, city_cfg) in enumerate(items, 1):
        logger.info(f"[{i:2d}/{len(items)}] {city_name}")
        df = fetch_climo_city(city_name, city_cfg)
        if not df.empty:
            all_dfs.append(df)
        if i < len(items):
            time.sleep(3)

    if not all_dfs:
        raise RuntimeError("No climo data returned")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(output_path, index=False)
    logger.info(f"Saved {len(combined)} rows → {output_path}")
    return combined


def add_climo_features(df: pd.DataFrame, climo_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Add climo_high, climo_low, anomaly_high, anomaly_low to a DataFrame
    that has 'city' and 'forecast_date' (or 'date') columns.
    Works for both training data and live feature matrix.
    """
    if climo_path is None:
        climo_path = CLIMO_PATH

    if not Path(climo_path).exists():
        logger.warning(f"Climo normals not found at {climo_path} — skipping climo features")
        return df

    climo = pd.read_csv(climo_path)

    df = df.copy()
    date_col = "forecast_date" if "forecast_date" in df.columns else "date"
    df["_doy"] = pd.to_datetime(df[date_col]).dt.dayofyear

    df = df.merge(climo, left_on=["city", "_doy"], right_on=["city", "doy"], how="left")
    df = df.drop(columns=["_doy", "doy"], errors="ignore")

    # Anomaly: how far is today's forecast from the climatological normal?
    # Use era5 for training data, ensemble mean for live inference
    high_col = "era5_high" if "era5_high" in df.columns else "ens_high_mean"
    low_col  = "era5_low"  if "era5_low"  in df.columns else "ens_low_mean"

    if high_col in df.columns:
        df["anomaly_high"] = df[high_col] - df["climo_high"]
    if low_col in df.columns:
        df["anomaly_low"]  = df[low_col]  - df["climo_low"]

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df = build_climo_normals()
    print(df.groupby("city")[["climo_high", "climo_low"]].mean().round(1).to_string())
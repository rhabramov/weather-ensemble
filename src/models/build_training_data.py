"""
One-time script to build the historical training dataset.

Joins IEM ASOS actuals to ERA5 reanalysis features, then adds
climatological normal features (climo_high, climo_low, anomaly).

Run once before training:
  python src/models/build_training_data.py
"""

import logging
import math
import time
from datetime import date, timedelta
from typing import Optional

import httpx
import pandas as pd

from config.cities import CITIES
from config.settings import GEFS_MEMBERS, ICON_MEMBERS, PROCESSED_DIR
from src.features.climo_features import add_climo_features, build_climo_normals, CLIMO_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ERA5_URL = "https://archive-api.open-meteo.com/v1/archive"


def get_temporal_features(d: date) -> dict:
    import math
    doy   = d.timetuple().tm_yday
    month = d.month
    return {
        "doy_sin":   math.sin(2 * math.pi * doy / 365),
        "doy_cos":   math.cos(2 * math.pi * doy / 365),
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
    }


def fetch_era5_city(city_name, city_cfg, start_date, end_date, max_retries=3):
    params = {
        "latitude":         city_cfg["lat"],
        "longitude":        city_cfg["lon"],
        "daily":            "temperature_2m_max,temperature_2m_min",
        "temperature_unit": "fahrenheit",
        "timezone":         city_cfg["tz"],
        "start_date":       start_date,
        "end_date":         end_date,
    }
    for attempt in range(1, max_retries + 1):
        try:
            r = httpx.get(ERA5_URL, params=params, timeout=60)
            if r.status_code == 429:
                wait = 30 * attempt
                logger.warning(f"429 for {city_name} — waiting {wait}s (attempt {attempt})")
                time.sleep(wait)
                continue
            r.raise_for_status()
            daily = r.json().get("daily", {})
            df = pd.DataFrame({
                "date":      pd.to_datetime(daily.get("time", [])),
                "city":      city_name,
                "era5_high": daily.get("temperature_2m_max", []),
                "era5_low":  daily.get("temperature_2m_min", []),
            })
            logger.info(f"  ERA5 {city_name}: {len(df)} days")
            return df
        except Exception as e:
            logger.error(f"  ERA5 failed for {city_name} (attempt {attempt}): {e}")
            if attempt < max_retries:
                time.sleep(15)
    return pd.DataFrame()


def fetch_all_era5(start_date, end_date):
    all_dfs = []
    items = list(CITIES.items())
    for i, (city_name, city_cfg) in enumerate(items, 1):
        logger.info(f"[{i:2d}/{len(items)}] Fetching ERA5 for {city_name}...")
        df = fetch_era5_city(city_name, city_cfg, start_date, end_date)
        if not df.empty:
            all_dfs.append(df)
        if i < len(items):
            time.sleep(3)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def build_training_data(
    actuals_path: Optional[str] = None,
    start_year: int = 2010,
    output_path: Optional[str] = None,
) -> pd.DataFrame:

    if actuals_path is None:
        actuals_path = str(PROCESSED_DIR / "historical_labels.csv")
    if output_path is None:
        output_path  = str(PROCESSED_DIR / "training_data.csv")

    # Load actuals
    actuals = pd.read_csv(actuals_path, parse_dates=["date"])
    actuals = actuals[actuals["date"].dt.year >= start_year]
    logger.info(f"Loaded {len(actuals)} actuals | cities: {actuals['city'].nunique()}")

    # Fetch ERA5 sequentially
    start_str = f"{start_year}-01-01"
    end_str   = str(date.today() - timedelta(days=2))
    logger.info(f"Fetching ERA5 from {start_str} to {end_str}...")
    era5_df = fetch_all_era5(start_str, end_str)
    if era5_df.empty:
        raise RuntimeError("ERA5 returned no data")

    # Join
    era5_df["date"] = pd.to_datetime(era5_df["date"])
    merged = actuals.merge(era5_df, on=["city", "date"], how="inner")
    logger.info(f"Merged: {len(merged)} rows | cities: {merged['city'].nunique()}")

    # Ensure climo normals exist, build if not
    if not CLIMO_PATH.exists():
        logger.info("Climo normals not found — building now (takes ~5 min)...")
        build_climo_normals()

    # Add climo features
    logger.info("Adding climatological normal features...")
    merged = add_climo_features(merged)
    logger.info(f"Climo columns added: {[c for c in merged.columns if 'climo' in c or 'anomaly' in c]}")

    # Proxy all forecast members with ERA5
    for col in ["gfs", "nam", "hrrr", "nws_official", "nws_hourly",
                "tomorrow", "wapi", "pirate"]:
        merged[f"{col}_high"] = merged["era5_high"]
        merged[f"{col}_low"]  = merged["era5_low"]

    for m in GEFS_MEMBERS:
        merged[f"gefs_m{m:02d}_high"] = merged["era5_high"]
        merged[f"gefs_m{m:02d}_low"]  = merged["era5_low"]

    for m in ICON_MEMBERS:
        merged[f"icon_m{m:02d}_high"] = merged["era5_high"]
        merged[f"icon_m{m:02d}_low"]  = merged["era5_low"]

    # Ensemble stats
    for target in ["high", "low"]:
        col = f"era5_{target}"
        merged[f"ens_{target}_mean"]      = merged[col]
        merged[f"ens_{target}_median"]    = merged[col]
        merged[f"ens_{target}_std"]       = 0.0
        merged[f"ens_{target}_spread"]    = 0.0
        merged[f"ens_{target}_p10"]       = merged[col]
        merged[f"ens_{target}_p25"]       = merged[col]
        merged[f"ens_{target}_p75"]       = merged[col]
        merged[f"ens_{target}_p90"]       = merged[col]
        merged[f"ens_{target}_n_members"] = 1

    merged["nws_vs_ens_high_delta"] = 0.0
    merged["nws_vs_ens_low_delta"]  = 0.0

    # Temporal features
    temporal = merged["date"].apply(lambda d: pd.Series(get_temporal_features(d.date())))
    merged   = pd.concat([merged, temporal], axis=1)

    # City index
    city_list = list(CITIES.keys())
    merged["city_idx"] = merged["city"].apply(
        lambda c: city_list.index(c) if c in city_list else -1
    )

    merged = merged.rename(columns={"date": "forecast_date"})
    merged["forecast_date"] = merged["forecast_date"].dt.strftime("%Y-%m-%d")

    merged.to_csv(output_path, index=False)
    logger.info(f"Saved {len(merged):,} rows × {merged.shape[1]} cols → {output_path}")
    logger.info(f"Cities: {merged['city'].nunique()}/20")
    logger.info(f"Date range: {merged['forecast_date'].min()} to {merged['forecast_date'].max()}")

    # Quick sanity check on new features
    if "anomaly_high" in merged.columns:
        logger.info(f"\nAnomaly stats (era5 vs climo):")
        logger.info(f"  anomaly_high: mean={merged['anomaly_high'].mean():.2f}°F  "
                    f"std={merged['anomaly_high'].std():.2f}°F")
        logger.info(f"  anomaly_low:  mean={merged['anomaly_low'].mean():.2f}°F  "
                    f"std={merged['anomaly_low'].std():.2f}°F")

    return merged


if __name__ == "__main__":
    df = build_training_data()
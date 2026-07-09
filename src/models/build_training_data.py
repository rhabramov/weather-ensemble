"""
One-time script to build the historical training dataset.

This is NOT run on the daily schedule. Run it once (takes ~1-2 hours)
to assemble the labeled training data before initial model training.

What it does:
  1. Loads historical actuals from NOAA CDO (already fetched + saved to CSV)
  2. For each historical date, reconstructs what the forecast features WOULD
     have been — since we can't re-call live APIs for the past, we use
     Open-Meteo's historical weather API (ERA5 reanalysis) as a proxy for
     what the models would have predicted.
  3. Joins features to actuals to create a labeled training DataFrame.
  4. Saves to data/processed/training_data.csv.

NOTE: Open-Meteo's ERA5 historical endpoint gives us reanalysis-based
high/low for any past date, which is a good stand-in for what GFS/NAM
would have forecast. It's not identical to a live forecast from 2018,
but it captures the signal. Once you have 6+ months of live predictions
saved in the DB, switch to using those as your training data instead.
"""

import asyncio
import logging
import math
from datetime import date, timedelta
from typing import Optional

import httpx
import pandas as pd

from config.cities import CITIES
from config.settings import GEFS_MEMBERS, ICON_MEMBERS, PROCESSED_DIR

logger = logging.getLogger(__name__)

ERA5_URL = "https://archive-api.open-meteo.com/v1/archive"


def get_temporal_features(d: date) -> dict:
    doy = d.timetuple().tm_yday
    month = d.month
    return {
        "doy_sin":   math.sin(2 * math.pi * doy / 365),
        "doy_cos":   math.cos(2 * math.pi * doy / 365),
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
    }


async def fetch_era5_for_city(
    client: httpx.AsyncClient,
    city_name: str,
    city_cfg: dict,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch ERA5 daily high/low for a city over a date range.
    Returns DataFrame with date, {city}_era5_high, {city}_era5_low.
    """
    params = {
        "latitude":         city_cfg["lat"],
        "longitude":        city_cfg["lon"],
        "daily":            "temperature_2m_max,temperature_2m_min",
        "temperature_unit": "fahrenheit",
        "timezone":         city_cfg["tz"],
        "start_date":       start_date,
        "end_date":         end_date,
    }
    try:
        resp = await client.get(ERA5_URL, params=params, timeout=60)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        df = pd.DataFrame({
            "date":      pd.to_datetime(daily.get("time", [])),
            "city":      city_name,
            "era5_high": daily.get("temperature_2m_max", []),
            "era5_low":  daily.get("temperature_2m_min", []),
        })
        logger.info(f"ERA5 {city_name}: {len(df)} days from {start_date} to {end_date}")
        return df
    except Exception as e:
        logger.error(f"ERA5 fetch failed for {city_name}: {e}")
        return pd.DataFrame()


async def fetch_all_era5(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch ERA5 for all cities and concatenate."""
    async with httpx.AsyncClient() as client:
        tasks = [
            fetch_era5_for_city(client, city_name, city_cfg, start_date, end_date)
            for city_name, city_cfg in CITIES.items()
        ]
        # Batch to avoid hammering Open-Meteo
        all_dfs = []
        for i in range(0, len(tasks), 5):
            batch = tasks[i:i+5]
            results = await asyncio.gather(*batch)
            all_dfs.extend([r for r in results if not r.empty])
            await asyncio.sleep(1)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def build_training_data(
    actuals_path: Optional[str] = None,
    start_year: int = 2015,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Join ERA5 historical features to NOAA CDO actuals.

    For the training set, ERA5 values serve as feature proxies for:
      - gfs_high / gfs_low (era5 reanalysis is close to what GFS would show)
      - nam_high / nam_low
      - hrrr_high / hrrr_low
      - nws_official_high / nws_official_low

    We use the same value for all since we can't recover 2015-era live
    forecasts. The ensemble stats will reflect this (zero spread for
    historical records), which is why once you have live predictions
    accumulating in the DB (6+ months), you should retrain on those.
    """
    if actuals_path is None:
        actuals_path = str(PROCESSED_DIR / "historical_labels.csv")
    if output_path is None:
        output_path = str(PROCESSED_DIR / "training_data.csv")

    # Load actuals
    actuals = pd.read_csv(actuals_path, parse_dates=["date"])
    actuals = actuals[actuals["date"].dt.year >= start_year]
    logger.info(f"Loaded {len(actuals)} actuals from {actuals_path}")

    # Fetch ERA5 for the same range
    start_str = f"{start_year}-01-01"
    end_str   = str(date.today() - timedelta(days=2))  # ERA5 lags ~2 days

    logger.info(f"Fetching ERA5 from {start_str} to {end_str}...")
    era5_df = asyncio.run(fetch_all_era5(start_str, end_str))

    if era5_df.empty:
        raise RuntimeError("ERA5 fetch returned no data")

    # Join actuals to ERA5 on city + date
    era5_df["date"] = pd.to_datetime(era5_df["date"])
    merged = actuals.merge(era5_df, on=["city", "date"], how="inner")
    logger.info(f"Merged: {len(merged)} rows after joining ERA5 to actuals")

    # Use ERA5 as proxy for all model members (they'll all be the same value)
    # This is the main limitation of historical training — live data is better
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

    # Ensemble stats (all same value → zero spread for historical, non-zero for live)
    merged["ens_high_mean"]   = merged["era5_high"]
    merged["ens_high_median"] = merged["era5_high"]
    merged["ens_high_std"]    = 0.0
    merged["ens_high_spread"] = 0.0
    merged["ens_high_p10"]    = merged["era5_high"]
    merged["ens_high_p25"]    = merged["era5_high"]
    merged["ens_high_p75"]    = merged["era5_high"]
    merged["ens_high_p90"]    = merged["era5_high"]
    merged["ens_high_n_members"] = 1

    merged["ens_low_mean"]   = merged["era5_low"]
    merged["ens_low_median"] = merged["era5_low"]
    merged["ens_low_std"]    = 0.0
    merged["ens_low_spread"] = 0.0
    merged["ens_low_p10"]    = merged["era5_low"]
    merged["ens_low_p25"]    = merged["era5_low"]
    merged["ens_low_p75"]    = merged["era5_low"]
    merged["ens_low_p90"]    = merged["era5_low"]
    merged["ens_low_n_members"] = 1

    merged["nws_vs_ens_high_delta"] = 0.0
    merged["nws_vs_ens_low_delta"]  = 0.0

    # Temporal features
    temporal = merged["date"].apply(lambda d: pd.Series(get_temporal_features(d.date())))
    merged = pd.concat([merged, temporal], axis=1)

    # City index
    city_list = list(CITIES.keys())
    merged["city_idx"] = merged["city"].apply(lambda c: city_list.index(c) if c in city_list else -1)

    # Rename for model compatibility
    merged = merged.rename(columns={"date": "forecast_date", "actual_high": "actual_high", "actual_low": "actual_low"})
    merged["forecast_date"] = merged["forecast_date"].dt.strftime("%Y-%m-%d")

    merged.to_csv(output_path, index=False)
    logger.info(f"Training data saved: {len(merged)} rows × {merged.shape[1]} cols → {output_path}")
    return merged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = build_training_data()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['forecast_date'].min()} to {df['forecast_date'].max()}")
    print(f"Cities: {df['city'].nunique()}")

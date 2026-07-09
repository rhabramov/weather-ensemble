"""
Feature matrix builder.

Takes the raw outputs from all ingestion modules and assembles a single
DataFrame row per city with all 34 forecast members plus derived
ensemble statistics. This is the input to the XGBoost model.

Feature groups:
  [1-2]   NWS official daily high/low
  [3-4]   NWS hourly-derived high/low
  [5-6]   GFS high/low
  [7-8]   NAM high/low
  [9-10]  HRRR high/low
  [11-26] GEFS members 0-15 (high + low = 32 values; listed as member features)
  [27-32] ICON members 0-5 (high + low = 12 values)
  [33]    Tomorrow.io high/low
  [34]    WeatherAPI high/low
  [35]    Pirate Weather high/low
  [+]     Ensemble statistics: mean, median, std, p10, p25, p75, p90, spread
  [+]     Temporal features: day_of_year_sin/cos, month_sin/cos
  [+]     City encoding (label-encoded categorical)
"""

import asyncio
import logging
import math
from datetime import datetime, date
from zoneinfo import ZoneInfo
from typing import Optional

import pandas as pd
import numpy as np

from config.cities import CITIES
from config.settings import GEFS_MEMBERS, ICON_MEMBERS
from src.ingestion.nws_forecast import fetch_all_nws_forecasts
from src.ingestion.open_meteo import fetch_all_open_meteo
from src.ingestion.third_party import fetch_all_third_party

logger = logging.getLogger(__name__)

CITY_LIST = list(CITIES.keys())
CITY_TO_IDX = {c: i for i, c in enumerate(CITY_LIST)}


def get_temporal_features(target_date: date) -> dict:
    """
    Sine/cosine encoding of day-of-year and month.
    Cyclical encoding means Jan 1 and Dec 31 are adjacent in feature space.
    """
    doy = target_date.timetuple().tm_yday
    month = target_date.month
    return {
        "doy_sin":   math.sin(2 * math.pi * doy / 365),
        "doy_cos":   math.cos(2 * math.pi * doy / 365),
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
    }


def ensemble_stats(values: list[Optional[float]], prefix: str) -> dict:
    """
    Compute ensemble statistics from a list of model values.
    Nones are excluded. Returns prefixed dict of stats.
    """
    clean = [v for v in values if v is not None]
    if not clean:
        return {
            f"{prefix}_mean":   None,
            f"{prefix}_median": None,
            f"{prefix}_std":    None,
            f"{prefix}_p10":    None,
            f"{prefix}_p25":    None,
            f"{prefix}_p75":    None,
            f"{prefix}_p90":    None,
            f"{prefix}_spread": None,
            f"{prefix}_n_members": 0,
        }
    arr = np.array(clean)
    return {
        f"{prefix}_mean":      float(np.mean(arr)),
        f"{prefix}_median":    float(np.median(arr)),
        f"{prefix}_std":       float(np.std(arr)),
        f"{prefix}_p10":       float(np.percentile(arr, 10)),
        f"{prefix}_p25":       float(np.percentile(arr, 25)),
        f"{prefix}_p75":       float(np.percentile(arr, 75)),
        f"{prefix}_p90":       float(np.percentile(arr, 90)),
        f"{prefix}_spread":    float(np.max(arr) - np.min(arr)),
        f"{prefix}_n_members": len(clean),
    }


def build_city_row(
    city_name: str,
    nws_data: dict,
    om_data: dict,
    tp_data: dict,
    target_date: date,
) -> dict:
    """
    Assemble a single feature row for one city from all data sources.
    """
    row = {"city": city_name, "city_idx": CITY_TO_IDX[city_name], "forecast_date": str(target_date)}
    row.update(get_temporal_features(target_date))

    # --- NWS official and hourly ---
    row["nws_official_high"] = nws_data.get("nws_official_high")
    row["nws_official_low"]  = nws_data.get("nws_official_low")
    row["nws_hourly_high"]   = nws_data.get("nws_hourly_high")
    row["nws_hourly_low"]    = nws_data.get("nws_hourly_low")

    # --- Deterministic models from Open-Meteo ---
    row["gfs_high"]  = om_data.get("gfs_high")
    row["gfs_low"]   = om_data.get("gfs_low")
    row["nam_high"]  = om_data.get("nam_high")
    row["nam_low"]   = om_data.get("nam_low")
    row["hrrr_high"] = om_data.get("hrrr_high")
    row["hrrr_low"]  = om_data.get("hrrr_low")

    # --- GEFS members ---
    gefs_highs, gefs_lows = [], []
    for m in GEFS_MEMBERS:
        h = om_data.get(f"gefs_m{m:02d}_high")
        l = om_data.get(f"gefs_m{m:02d}_low")
        row[f"gefs_m{m:02d}_high"] = h
        row[f"gefs_m{m:02d}_low"]  = l
        gefs_highs.append(h)
        gefs_lows.append(l)

    # --- ICON members ---
    icon_highs, icon_lows = [], []
    for m in ICON_MEMBERS:
        h = om_data.get(f"icon_m{m:02d}_high")
        l = om_data.get(f"icon_m{m:02d}_low")
        row[f"icon_m{m:02d}_high"] = h
        row[f"icon_m{m:02d}_low"]  = l
        icon_highs.append(h)
        icon_lows.append(l)

    # --- Third-party ---
    row["tomorrow_high"] = tp_data.get("tomorrow_high")
    row["tomorrow_low"]  = tp_data.get("tomorrow_low")
    row["wapi_high"]     = tp_data.get("wapi_high")
    row["wapi_low"]      = tp_data.get("wapi_low")
    row["pirate_high"]   = tp_data.get("pirate_high")
    row["pirate_low"]    = tp_data.get("pirate_low")

    # --- All-member ensemble stats ---
    all_highs = (
        [row["nws_official_high"], row["nws_hourly_high"],
         row["gfs_high"], row["nam_high"], row["hrrr_high"]]
        + gefs_highs + icon_highs
        + [row["tomorrow_high"], row["wapi_high"], row["pirate_high"]]
    )
    all_lows = (
        [row["nws_official_low"], row["nws_hourly_low"],
         row["gfs_low"], row["nam_low"], row["hrrr_low"]]
        + gefs_lows + icon_lows
        + [row["tomorrow_low"], row["wapi_low"], row["pirate_low"]]
    )

    row.update(ensemble_stats(all_highs, "ens_high"))
    row.update(ensemble_stats(all_lows,  "ens_low"))

    # NWS deviation from ensemble mean (human forecaster divergence signal)
    if row["nws_official_high"] is not None and row["ens_high_mean"] is not None:
        row["nws_vs_ens_high_delta"] = row["nws_official_high"] - row["ens_high_mean"]
    else:
        row["nws_vs_ens_high_delta"] = None

    if row["nws_official_low"] is not None and row["ens_low_mean"] is not None:
        row["nws_vs_ens_low_delta"] = row["nws_official_low"] - row["ens_low_mean"]
    else:
        row["nws_vs_ens_low_delta"] = None

    return row


async def build_feature_matrix(target_date: Optional[date] = None) -> pd.DataFrame:
    """
    Fetch all data sources concurrently and return a 20-row DataFrame
    (one per city) with all features assembled.

    target_date defaults to today.
    """
    if target_date is None:
        target_date = date.today()

    logger.info(f"Building feature matrix for {target_date} across {len(CITIES)} cities...")

    # Fetch all sources concurrently
    nws_list, om_list, tp_list = await asyncio.gather(
        fetch_all_nws_forecasts(),
        fetch_all_open_meteo(),
        fetch_all_third_party(),
    )

    # Index by city for O(1) lookup
    nws_by_city = {r["city"]: r for r in nws_list}
    om_by_city  = {r["city"]: r for r in om_list}
    tp_by_city  = {r["city"]: r for r in tp_list}

    rows = []
    for city_name in CITIES:
        row = build_city_row(
            city_name,
            nws_by_city.get(city_name, {}),
            om_by_city.get(city_name, {}),
            tp_by_city.get(city_name, {}),
            target_date,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"Feature matrix built: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def get_feature_columns(df: pd.DataFrame, target: str = "high") -> list[str]:
    """
    Return the list of numeric feature columns for model training/inference.
    Excludes metadata columns and the opposite target.
    """
    exclude = {
        "city", "forecast_date",
        "actual_high", "actual_low",  # labels — never in features
        "actual_high_temp", "actual_low_temp",
    }
    cols = [c for c in df.columns if c not in exclude]
    return cols


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = asyncio.run(build_feature_matrix())
    print(df[["city", "ens_high_mean", "ens_high_std", "ens_low_mean", "nws_official_high"]].to_string())
    print(f"\nTotal features: {df.shape[1]}")

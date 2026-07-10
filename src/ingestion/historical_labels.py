"""
Historical label collection via Iowa Environmental Mesonet (IEM) daily endpoint.

Uses state-specific network codes (e.g. MA_ASOS, NY_ASOS) which the IEM
daily.py endpoint requires. Fetches one city at a time sequentially to
avoid 429 rate limiting.

Free, no API key required.

Run once before training:
  python src/ingestion/historical_labels.py
"""

import asyncio
import logging
import io
import time
from datetime import date, timedelta
from typing import Optional

import httpx
import pandas as pd

from config.settings import PROCESSED_DIR

logger = logging.getLogger(__name__)

IEM_DAILY_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py"

# State-specific IEM network codes + ASOS station IDs
# Network format is {STATE_ABBREV}_ASOS
STATIONS = {
    "seattle":        {"network": "WA_ASOS", "station": "SEA"},
    "los_angeles":    {"network": "CA_ASOS", "station": "LAX"},
    "miami":          {"network": "FL_ASOS", "station": "MIA"},
    "new_york":       {"network": "NY_ASOS", "station": "JFK"},
    "minneapolis":    {"network": "MN_ASOS", "station": "MSP"},
    "houston":        {"network": "TX_ASOS", "station": "HOU"},
    "denver":         {"network": "CO_ASOS", "station": "DEN"},
    "boston":         {"network": "MA_ASOS", "station": "BOS"},
    "chicago":        {"network": "IL_ASOS", "station": "MDW"},
    "dallas":         {"network": "TX_ASOS", "station": "DFW"},
    "philadelphia":   {"network": "PA_ASOS", "station": "PHL"},
    "san_francisco":  {"network": "CA_ASOS", "station": "SFO"},
    "las_vegas":      {"network": "NV_ASOS", "station": "LAS"},
    "oklahoma_city":  {"network": "OK_ASOS", "station": "OKC"},
    "austin":         {"network": "TX_ASOS", "station": "AUS"},
    "san_antonio":    {"network": "TX_ASOS", "station": "SAT"},
    "phoenix":        {"network": "AZ_ASOS", "station": "PHX"},
    "new_orleans":    {"network": "LA_ASOS", "station": "MSY"},
    "atlanta":        {"network": "GA_ASOS", "station": "ATL"},
    "washington_dc":  {"network": "VA_ASOS", "station": "DCA"},
}


def fetch_station_sync(
    city_name: str,
    network: str,
    station: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Synchronous fetch for one station. Returns DataFrame with
    city, date, actual_high, actual_low — all in Fahrenheit.
    """
    params = {
        "network":  network,
        "stations": station,
        "year1":    start_date[:4],
        "month1":   start_date[5:7],
        "day1":     start_date[8:10],
        "year2":    end_date[:4],
        "month2":   end_date[5:7],
        "day2":     end_date[8:10],
        "vars":     "max_temp_f,min_temp_f",
        "what":     "download",
        "delim":    "comma",
        "gis":      "no",
    }

    try:
        r = httpx.get(IEM_DAILY_URL, params=params, timeout=60)
        r.raise_for_status()
        raw = r.text.strip()

        if raw.startswith("ERROR") or not raw:
            logger.error(f"IEM error for {city_name} ({network}/{station}): {raw[:100]}")
            return pd.DataFrame()

        lines = [l for l in raw.splitlines() if l and not l.startswith("#")]
        if len(lines) < 2:
            logger.warning(f"IEM no data rows for {city_name}")
            return pd.DataFrame()

        df = pd.read_csv(io.StringIO("\n".join(lines)))
        df.columns = [c.strip().lower() for c in df.columns]

        # Log columns so we can debug if format changes
        logger.debug(f"IEM columns for {city_name}: {df.columns.tolist()}")
        logger.debug(f"IEM first row for {city_name}: {df.iloc[0].to_dict() if not df.empty else 'empty'}")

        if "max_temp_f" not in df.columns or "min_temp_f" not in df.columns:
            logger.error(f"Missing temp columns for {city_name}. Got: {df.columns.tolist()}")
            return pd.DataFrame()

        # Date column is called 'day' in IEM daily output
        date_col = "day" if "day" in df.columns else df.columns[1]
        df = df.rename(columns={
            date_col:    "date",
            "max_temp_f": "actual_high",
            "min_temp_f": "actual_low",
        })

        df["city"]        = city_name
        df["date"]        = pd.to_datetime(df["date"], errors="coerce")
        df["actual_high"] = pd.to_numeric(df["actual_high"], errors="coerce")
        df["actual_low"]  = pd.to_numeric(df["actual_low"],  errors="coerce")
        df = df.dropna(subset=["date", "actual_high", "actual_low"])

        logger.info(f"  {city_name:20s} ({station}): {len(df):4d} days  "
                    f"high range {df['actual_high'].min():.0f}–{df['actual_high'].max():.0f}°F  "
                    f"low range {df['actual_low'].min():.0f}–{df['actual_low'].max():.0f}°F")

        return df[["city", "date", "actual_high", "actual_low"]]

    except Exception as e:
        logger.error(f"IEM fetch failed for {city_name}: {e}")
        return pd.DataFrame()


def fetch_all_historical(
    start_year: int = 2015,
    end_date: Optional[str] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch IEM ASOS daily high/low for all 20 cities sequentially.
    Sequential (not async) to avoid IEM rate limiting.
    """
    if end_date is None:
        end_date = str(date.today() - timedelta(days=1))
    start_date = f"{start_year}-01-01"

    if output_path is None:
        output_path = str(PROCESSED_DIR / "historical_labels.csv")

    logger.info(f"Fetching IEM ASOS for {len(STATIONS)} cities: {start_date} to {end_date}")
    logger.info("Running sequentially (1 city at a time) to respect IEM rate limits...")

    all_dfs = []
    for i, (city_name, cfg) in enumerate(STATIONS.items(), 1):
        logger.info(f"[{i:2d}/{len(STATIONS)}] Fetching {city_name}...")
        df = fetch_station_sync(
            city_name, cfg["network"], cfg["station"], start_date, end_date
        )
        if not df.empty:
            all_dfs.append(df)
        else:
            logger.error(f"  FAILED: {city_name} — no data returned")

        # Pause between requests to avoid rate limiting
        if i < len(STATIONS):
            time.sleep(1.0)

    if not all_dfs:
        raise RuntimeError("IEM returned no data for any city. Check network codes above.")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.dropna(subset=["actual_high", "actual_low"])
    combined = combined.sort_values(["city", "date"]).reset_index(drop=True)

    # Spot checks against known NWS CLI values
    logger.info("\n=== Spot checks vs known NWS CLI values ===")
    checks = [
        ("boston",        "2015-01-01", 28,  10),   # polar air mass
        ("phoenix",       "2020-07-15", 112, 91),   # extreme heat event
        ("minneapolis",   "2019-01-30", -6,  -21),  # polar vortex
        ("miami",         "2018-08-01", 91,  80),   # typical summer
        ("new_york",      "2016-01-23", 26,  11),   # Jonas blizzard
    ]
    all_ok = True
    for city, check_date, exp_high, exp_low in checks:
        row = combined[
            (combined["city"] == city) &
            (combined["date"].dt.strftime("%Y-%m-%d") == check_date)
        ]
        if not row.empty:
            hi = row.iloc[0]["actual_high"]
            lo = row.iloc[0]["actual_low"]
            hi_ok = abs(hi - exp_high) <= 4
            lo_ok = abs(lo - exp_low)  <= 4
            status = "✓" if (hi_ok and lo_ok) else "✗"
            if not (hi_ok and lo_ok):
                all_ok = False
            logger.info(
                f"  {status} {city:15s} {check_date}  "
                f"high={hi:.0f}°F (exp ~{exp_high})  "
                f"low={lo:.0f}°F (exp ~{exp_low})"
            )
        else:
            logger.warning(f"  ? {city} {check_date}: no data")

    if all_ok:
        logger.info("All spot checks passed ✓")
    else:
        logger.warning("Some spot checks failed — review values above before training")

    combined.to_csv(output_path, index=False)
    logger.info(f"\nSaved {len(combined):,} rows → {output_path}")
    logger.info(f"Date range: {combined['date'].min().date()} to {combined['date'].max().date()}")
    logger.info(f"Cities: {combined['city'].nunique()}/20")

    print("\n=== Per-city mean high / mean low (°F) ===")
    print(combined.groupby("city")[["actual_high", "actual_low"]].mean().round(1).to_string())

    return combined


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    df = fetch_all_historical(start_year=2010)
"""
Historical label collection via NOAA Climate Data Online (CDO) API.

Used once (or periodically) to build the training dataset.
Fetches TMAX and TMIN (daily high/low in tenths of degrees C, converted to F)
for each city's GHCND station going back 10 years.

Requires a free NOAA CDO API key: https://www.ncdc.noaa.gov/cdo-web/token

Rate limit: 1000 requests/day, 5 requests/second. We handle this with
throttling and batching by year.

GHCND station IDs for our cities (airport stations):
"""

import asyncio
import logging
import time
from datetime import date, timedelta
from typing import Optional

import httpx
import pandas as pd

from config.settings import NOAA_CDO_KEY, PROCESSED_DIR

logger = logging.getLogger(__name__)

CDO_BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"

# GHCND station IDs mapped to our city keys
GHCND_STATIONS = {
    "seattle":        "GHCND:USW00024233",
    "los_angeles":    "GHCND:USW00023174",
    "miami":          "GHCND:USW00012839",
    "new_york":       "GHCND:USW00094789",  # JFK
    "minneapolis":    "GHCND:USW00014922",
    "houston":        "GHCND:USW00012960",
    "denver":         "GHCND:USW00003017",
    "boston":         "GHCND:USW00014739",
    "chicago":        "GHCND:USW00094846",  # MDW
    "dallas":         "GHCND:USW00003927",
    "philadelphia":   "GHCND:USW00013739",
    "san_francisco":  "GHCND:USW00023234",
    "las_vegas":      "GHCND:USW00023169",
    "oklahoma_city":  "GHCND:USW00013967",
    "austin":         "GHCND:USW00013958",
    "san_antonio":    "GHCND:USW00012921",
    "phoenix":        "GHCND:USW00023183",
    "new_orleans":    "GHCND:USW00012916",
    "atlanta":        "GHCND:USW00013874",
    "washington_dc":  "GHCND:USW00013743",
}


def tenths_c_to_f(tenths_c: Optional[float]) -> Optional[float]:
    """NOAA CDO returns temps in tenths of degrees Celsius."""
    if tenths_c is None:
        return None
    return round(tenths_c / 10 * 9 / 5 + 32, 1)


async def fetch_station_year(
    client: httpx.AsyncClient,
    city: str,
    station_id: str,
    year: int,
) -> list[dict]:
    """
    Fetch TMAX and TMIN for one station for one calendar year.
    CDO API max date range per request is 1 year.
    """
    params = {
        "datasetid":  "GHCND",
        "stationid":  station_id,
        "datatypeid": "TMAX,TMIN",
        "startdate":  f"{year}-01-01",
        "enddate":    f"{year}-12-31",
        "limit":      1000,
        "units":      "metric",  # returns tenths of C
    }
    headers = {"token": NOAA_CDO_KEY}

    records = []
    try:
        resp = await client.get(CDO_BASE, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("results", [])

        # Pivot: each date gets one TMAX and one TMIN entry as separate records
        by_date: dict[str, dict] = {}
        for rec in raw:
            d = rec["date"][:10]  # YYYY-MM-DD
            dtype = rec["datatype"]
            val = rec.get("value")
            if d not in by_date:
                by_date[d] = {"city": city, "date": d, "actual_high": None, "actual_low": None}
            if dtype == "TMAX":
                by_date[d]["actual_high"] = tenths_c_to_f(val)
            elif dtype == "TMIN":
                by_date[d]["actual_low"] = tenths_c_to_f(val)

        records = list(by_date.values())
        logger.info(f"CDO {city} {year}: {len(records)} days fetched")

    except Exception as e:
        logger.error(f"CDO fetch failed for {city} {year}: {e}")

    return records


async def fetch_all_historical(
    start_year: int = 2015,
    end_year: Optional[int] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch 10+ years of daily high/low for all 20 cities.

    Saves to CSV at output_path (defaults to data/processed/historical_labels.csv).
    Rate limited to ~4 requests/second to stay within CDO limits.
    """
    if not NOAA_CDO_KEY:
        raise ValueError("NOAA_CDO_KEY not set in .env — get a free key at ncdc.noaa.gov/cdo-web/token")

    if end_year is None:
        end_year = date.today().year - 1  # full years only for training

    if output_path is None:
        output_path = str(PROCESSED_DIR / "historical_labels.csv")

    all_records = []

    async with httpx.AsyncClient() as client:
        for year in range(start_year, end_year + 1):
            logger.info(f"Fetching year {year}...")
            tasks = [
                fetch_station_year(client, city, station_id, year)
                for city, station_id in GHCND_STATIONS.items()
            ]
            # Throttle: process 4 cities at a time to respect rate limits
            for i in range(0, len(tasks), 4):
                batch = tasks[i:i+4]
                results = await asyncio.gather(*batch)
                for city_records in results:
                    all_records.extend(city_records)
                await asyncio.sleep(1.0)  # ~1s between batches = safe rate

    df = pd.DataFrame(all_records)
    df = df.dropna(subset=["actual_high", "actual_low"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} historical records to {output_path}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = asyncio.run(fetch_all_historical(start_year=2015))
    print(df.head(20))
    print(f"\nShape: {df.shape}")
    print(df.groupby("city")[["actual_high", "actual_low"]].describe())
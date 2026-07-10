"""
NWS Forecast API ingestion.

Pulls two products per city:
  1. Daily forecast (official NWS high/low — members 1 & 2)
  2. Hourly forecast (derive max/min for the calendar day — members 3 & 4)

Grid coordinates are looked up dynamically from the NWS /points/ API
on first use and cached in memory for the run — this avoids stale
hardcoded grids which 404 when NWS reorganizes their grid system.
"""

import logging
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional

import httpx

from config.cities import CITIES

logger = logging.getLogger(__name__)

NWS_POINTS_URL   = "https://api.weather.gov/points/{lat},{lon}"
NWS_FORECAST_URL = "https://api.weather.gov/gridpoints/{office}/{x},{y}/forecast"
NWS_HOURLY_URL   = "https://api.weather.gov/gridpoints/{office}/{x},{y}/forecast/hourly"
HEADERS = {"User-Agent": "WeatherEnsemble/1.0 (contact@yourdomain.com)"}

# In-memory cache: city_name -> {"office": str, "x": int, "y": int}
_grid_cache: dict[str, dict] = {}


async def get_grid(client: httpx.AsyncClient, city_name: str, city_cfg: dict) -> Optional[dict]:
    """
    Look up NWS grid office and coordinates from lat/lon via the /points/ API.
    Caches result in memory for the duration of the run.
    """
    if city_name in _grid_cache:
        return _grid_cache[city_name]

    url = NWS_POINTS_URL.format(lat=city_cfg["lat"], lon=city_cfg["lon"])
    try:
        resp = await client.get(url, timeout=15)
        resp.raise_for_status()
        props = resp.json()["properties"]
        grid = {
            "office": props["gridId"],
            "x":      props["gridX"],
            "y":      props["gridY"],
        }
        _grid_cache[city_name] = grid
        logger.info(f"NWS grid {city_name}: office={grid['office']} x={grid['x']} y={grid['y']}")
        return grid
    except Exception as e:
        logger.error(f"NWS /points/ lookup failed for {city_name}: {e}")
        return None


async def fetch_nws_daily(
    client: httpx.AsyncClient, city_name: str, city_cfg: dict
) -> dict:
    """Fetch official NWS daily forecast high/low for today."""
    result = {"city": city_name, "nws_official_high": None, "nws_official_low": None}

    grid = await get_grid(client, city_name, city_cfg)
    if not grid:
        return result

    url = NWS_FORECAST_URL.format(**grid)
    try:
        resp = await client.get(url, timeout=15)
        resp.raise_for_status()
        periods = resp.json()["properties"]["periods"]

        local_tz = ZoneInfo(city_cfg["tz"])
        today = datetime.now(local_tz).date()

        for period in periods:
            start = datetime.fromisoformat(period["startTime"]).astimezone(local_tz)
            if start.date() != today:
                continue
            temp = period.get("temperature")
            if period.get("isDaytime") and result["nws_official_high"] is None:
                result["nws_official_high"] = temp
            elif not period.get("isDaytime") and result["nws_official_low"] is None:
                result["nws_official_low"] = temp

    except Exception as e:
        logger.error(f"NWS daily forecast failed for {city_name}: {e}")

    logger.info(f"NWS daily {city_name}: high={result['nws_official_high']} low={result['nws_official_low']}")
    return result


async def fetch_nws_hourly(
    client: httpx.AsyncClient, city_name: str, city_cfg: dict
) -> dict:
    """Fetch NWS hourly forecast and derive today's max and min temps."""
    result = {"city": city_name, "nws_hourly_high": None, "nws_hourly_low": None}

    grid = await get_grid(client, city_name, city_cfg)
    if not grid:
        return result

    url = NWS_HOURLY_URL.format(**grid)
    try:
        resp = await client.get(url, timeout=15)
        resp.raise_for_status()
        periods = resp.json()["properties"]["periods"]

        local_tz = ZoneInfo(city_cfg["tz"])
        today = datetime.now(local_tz).date()

        today_temps = []
        for period in periods:
            start = datetime.fromisoformat(period["startTime"]).astimezone(local_tz)
            if start.date() == today:
                temp = period.get("temperature")
                if temp is not None:
                    today_temps.append(temp)

        if today_temps:
            result["nws_hourly_high"] = max(today_temps)
            result["nws_hourly_low"]  = min(today_temps)

    except Exception as e:
        logger.error(f"NWS hourly forecast failed for {city_name}: {e}")

    logger.info(f"NWS hourly {city_name}: high={result['nws_hourly_high']} low={result['nws_hourly_low']}")
    return result


async def fetch_city_nws(client: httpx.AsyncClient, city_name: str, city_cfg: dict) -> dict:
    """Fetch both daily and hourly NWS products for one city."""
    # Look up grid first (shared between daily and hourly)
    await get_grid(client, city_name, city_cfg)
    daily, hourly = await asyncio.gather(
        fetch_nws_daily(client, city_name, city_cfg),
        fetch_nws_hourly(client, city_name, city_cfg),
    )
    return {**daily, **hourly}


async def fetch_all_nws_forecasts() -> list[dict]:
    """Fetch NWS forecasts for all 20 cities concurrently."""
    async with httpx.AsyncClient(headers=HEADERS) as client:
        tasks = [
            fetch_city_nws(client, city_name, city_cfg)
            for city_name, city_cfg in CITIES.items()
        ]
        results = await asyncio.gather(*tasks)
    return list(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = asyncio.run(fetch_all_nws_forecasts())
    for r in results:
        print(r)
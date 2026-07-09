"""
NWS Forecast API ingestion.

Pulls two products per city:
  1. Daily forecast (official NWS high/low — members 1 & 2)
  2. Hourly forecast (derive max/min for the calendar day — members 3 & 4)

No API key required. NWS requests a User-Agent header with contact info.
"""

import logging
import asyncio
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional

import httpx

from config.cities import CITIES

logger = logging.getLogger(__name__)

NWS_FORECAST_URL = "https://api.weather.gov/gridpoints/{office}/{grid}/forecast"
NWS_HOURLY_URL   = "https://api.weather.gov/gridpoints/{office}/{grid}/forecast/hourly"
HEADERS = {"User-Agent": "WeatherEnsemble/1.0 (contact@yourdomain.com)"}


async def fetch_nws_daily(
    client: httpx.AsyncClient, city_name: str, city_cfg: dict
) -> dict:
    """
    Fetch official NWS daily forecast and extract today's high and low.

    The daily forecast periods alternate Day/Night. We grab the first
    'daytime' period (isDaytime=True) for the high and the following
    night period for the low.
    """
    url = NWS_FORECAST_URL.format(
        office=city_cfg["nws_office"], grid=city_cfg["nws_grid"]
    )
    result = {
        "city": city_name,
        "nws_official_high": None,
        "nws_official_low": None,
    }
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
    """
    Fetch NWS hourly forecast and derive today's max and min temps.

    This captures intraday resolution that the daily product averages over,
    giving us two independent members (members 3 & 4).
    """
    url = NWS_HOURLY_URL.format(
        office=city_cfg["nws_office"], grid=city_cfg["nws_grid"]
    )
    result = {
        "city": city_name,
        "nws_hourly_high": None,
        "nws_hourly_low": None,
    }
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
            result["nws_hourly_low"] = min(today_temps)

    except Exception as e:
        logger.error(f"NWS hourly forecast failed for {city_name}: {e}")

    logger.info(f"NWS hourly {city_name}: high={result['nws_hourly_high']} low={result['nws_hourly_low']}")
    return result


async def fetch_city_nws(client: httpx.AsyncClient, city_name: str, city_cfg: dict) -> dict:
    """Fetch both daily and hourly NWS products for one city and merge."""
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

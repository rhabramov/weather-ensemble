"""
Third-party forecast API ingestion.

Members pulled here:
  Member 33: Tomorrow.io daily high
  Member 33: Tomorrow.io daily low
  Member 34: WeatherAPI daily high
  Member 34: WeatherAPI daily low
  Member 35: Pirate Weather hourly-derived high
  Member 35: Pirate Weather hourly-derived low

All require API keys in .env. Failures are caught and return None — a
missing key means those members are excluded from the feature matrix
(handled gracefully in the feature builder).
"""

import logging
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional

import httpx

from config.cities import CITIES
from config.settings import TOMORROW_IO_KEY, WEATHER_API_KEY, PIRATE_WEATHER_KEY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tomorrow.io
# ---------------------------------------------------------------------------

async def fetch_tomorrow_io(
    client: httpx.AsyncClient, city_name: str, city_cfg: dict
) -> dict:
    """
    Fetch daily high/low from Tomorrow.io Timelines API.
    Free tier: 500 calls/day, 25 calls/hour.
    Docs: https://docs.tomorrow.io/reference/get-timelines
    """
    result = {"city": city_name, "tomorrow_high": None, "tomorrow_low": None}

    if not TOMORROW_IO_KEY:
        logger.warning("TOMORROW_IO_KEY not set — skipping Tomorrow.io")
        return result

    url = "https://api.tomorrow.io/v4/timelines"
    params = {
        "location":  f"{city_cfg['lat']},{city_cfg['lon']}",
        "fields":    "temperatureMax,temperatureMin",
        "units":     "imperial",
        "timesteps": "1d",
        "apikey":    TOMORROW_IO_KEY,
    }
    try:
        resp = await client.get(url, params=params, timeout=15)
        resp.raise_for_status()
        intervals = (
            resp.json()
            .get("data", {})
            .get("timelines", [{}])[0]
            .get("intervals", [])
        )
        local_tz = ZoneInfo(city_cfg["tz"])
        today = datetime.now(local_tz).date()

        for interval in intervals:
            start_str = interval.get("startTime", "")
            if not start_str:
                continue
            interval_date = datetime.fromisoformat(start_str).astimezone(local_tz).date()
            if interval_date == today:
                vals = interval.get("values", {})
                result["tomorrow_high"] = vals.get("temperatureMax")
                result["tomorrow_low"]  = vals.get("temperatureMin")
                break

    except Exception as e:
        logger.error(f"Tomorrow.io failed for {city_name}: {e}")

    logger.info(f"Tomorrow.io {city_name}: high={result['tomorrow_high']} low={result['tomorrow_low']}")
    return result


# ---------------------------------------------------------------------------
# WeatherAPI.com
# ---------------------------------------------------------------------------

async def fetch_weather_api(
    client: httpx.AsyncClient, city_name: str, city_cfg: dict
) -> dict:
    """
    Fetch daily high/low from WeatherAPI.com forecast endpoint.
    Free tier: 1M calls/month, 1 day forecast included.
    Docs: https://www.weatherapi.com/docs/
    """
    result = {"city": city_name, "wapi_high": None, "wapi_low": None}

    if not WEATHER_API_KEY:
        logger.warning("WEATHER_API_KEY not set — skipping WeatherAPI")
        return result

    url = "http://api.weatherapi.com/v1/forecast.json"
    params = {
        "key":  WEATHER_API_KEY,
        "q":    f"{city_cfg['lat']},{city_cfg['lon']}",
        "days": 1,
        "aqi":  "no",
        "alerts": "no",
    }
    try:
        resp = await client.get(url, params=params, timeout=15)
        resp.raise_for_status()
        forecast_day = (
            resp.json()
            .get("forecast", {})
            .get("forecastday", [{}])[0]
            .get("day", {})
        )
        result["wapi_high"] = forecast_day.get("maxtemp_f")
        result["wapi_low"]  = forecast_day.get("mintemp_f")

    except Exception as e:
        logger.error(f"WeatherAPI failed for {city_name}: {e}")

    logger.info(f"WeatherAPI {city_name}: high={result['wapi_high']} low={result['wapi_low']}")
    return result


# ---------------------------------------------------------------------------
# Pirate Weather
# ---------------------------------------------------------------------------

async def fetch_pirate_weather(
    client: httpx.AsyncClient, city_name: str, city_cfg: dict
) -> dict:
    """
    Fetch hourly forecast from Pirate Weather and derive today's high/low.
    Open-source reimplementation of the Dark Sky API.
    Free tier: 10K calls/month.
    Docs: https://pirateweather.net/en/latest/
    """
    result = {"city": city_name, "pirate_high": None, "pirate_low": None}

    if not PIRATE_WEATHER_KEY:
        logger.warning("PIRATE_WEATHER_KEY not set — skipping Pirate Weather")
        return result

    url = f"https://api.pirateweather.net/forecast/{PIRATE_WEATHER_KEY}/{city_cfg['lat']},{city_cfg['lon']}"
    params = {"units": "us", "exclude": "currently,minutely,daily,alerts"}

    try:
        resp = await client.get(url, params=params, timeout=15)
        resp.raise_for_status()
        hourly_data = resp.json().get("hourly", {}).get("data", [])

        local_tz = ZoneInfo(city_cfg["tz"])
        today = datetime.now(local_tz).date()

        today_temps = []
        for hour in hourly_data:
            ts = datetime.fromtimestamp(hour["time"], tz=local_tz)
            if ts.date() == today:
                temp = hour.get("temperature")
                if temp is not None:
                    today_temps.append(temp)

        if today_temps:
            result["pirate_high"] = max(today_temps)
            result["pirate_low"]  = min(today_temps)

    except Exception as e:
        logger.error(f"Pirate Weather failed for {city_name}: {e}")

    logger.info(f"Pirate Weather {city_name}: high={result['pirate_high']} low={result['pirate_low']}")
    return result


# ---------------------------------------------------------------------------
# Combined fetch
# ---------------------------------------------------------------------------

async def fetch_city_third_party(
    client: httpx.AsyncClient, city_name: str, city_cfg: dict
) -> dict:
    tomorrow, wapi, pirate = await asyncio.gather(
        fetch_tomorrow_io(client, city_name, city_cfg),
        fetch_weather_api(client, city_name, city_cfg),
        fetch_pirate_weather(client, city_name, city_cfg),
    )
    return {**tomorrow, **wapi, **pirate}


async def fetch_all_third_party() -> list[dict]:
    """Fetch all third-party forecasts for all 20 cities concurrently."""
    async with httpx.AsyncClient() as client:
        tasks = [
            fetch_city_third_party(client, city_name, city_cfg)
            for city_name, city_cfg in CITIES.items()
        ]
        results = await asyncio.gather(*tasks)
    return list(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = asyncio.run(fetch_all_third_party())
    for r in results:
        print(r)

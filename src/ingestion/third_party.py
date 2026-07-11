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

import asyncio
import logging
import random
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from config.cities import CITIES
from config.settings import TOMORROW_IO_KEY, WEATHER_API_KEY, PIRATE_WEATHER_KEY

logger = logging.getLogger(__name__)

TOMORROW_URL = "https://api.tomorrow.io/v4/timelines"
TOMORROW_MAX_RETRIES = 4
TOMORROW_CONCURRENCY = 2
TOMORROW_CITY_DELAY_SECONDS = 3.0 


def _retry_sleep_seconds(attempt: int, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    base = min(2 ** attempt, 60)
    return base * random.uniform(0.8, 1.3)


async def _get_with_retry_429(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict,
    timeout: float = 15,
    max_retries: int = 4,
    provider: str = "provider",
    city_name: str = "",
) -> httpx.Response | None:
    for attempt in range(max_retries + 1):
        try:
            resp = await client.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                if attempt == max_retries:
                    logger.warning("%s rate-limited for %s after %s retries", provider, city_name, max_retries)
                    return None
                sleep_s = _retry_sleep_seconds(attempt, resp.headers.get("Retry-After"))
                logger.warning(
                    "%s rate-limited for %s, retrying in %.2fs (%s/%s)",
                    provider, city_name, sleep_s, attempt + 1, max_retries
                )
                await asyncio.sleep(sleep_s)
                continue
            resp.raise_for_status()
            return resp
        except httpx.RequestError as e:
            if attempt == max_retries:
                logger.error("%s request error for %s: %s", provider, city_name, e)
                return None
            sleep_s = _retry_sleep_seconds(attempt)
            logger.warning(
                "%s request error for %s, retrying in %.2fs (%s/%s)",
                provider, city_name, sleep_s, attempt + 1, max_retries
            )
            await asyncio.sleep(sleep_s)
        except httpx.HTTPStatusError as e:
            logger.error("%s failed for %s: %s", provider, city_name, e)
            return None
    return None


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

    params = {
        "location": f"{city_cfg['lat']},{city_cfg['lon']}",
        "fields": "temperatureMax,temperatureMin",
        "units": "imperial",
        "timesteps": "1d",
        "apikey": TOMORROW_IO_KEY,
    }

    resp = await _get_with_retry_429(
        client,
        TOMORROW_URL,
        params=params,
        timeout=15,
        max_retries=TOMORROW_MAX_RETRIES,
        provider="Tomorrow.io",
        city_name=city_name,
    )
    if resp is None:
        logger.info("Tomorrow.io %s: high=None low=None", city_name)
        return result

    try:
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
                result["tomorrow_low"] = vals.get("temperatureMin")
                break
    except Exception as e:
        logger.error("Tomorrow.io failed for %s: %s", city_name, e)

    logger.info(
        "Tomorrow.io %s: high=%s low=%s",
        city_name,
        result["tomorrow_high"],
        result["tomorrow_low"],
    )
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
        "key": WEATHER_API_KEY,
        "q": f"{city_cfg['lat']},{city_cfg['lon']}",
        "days": 1,
        "aqi": "no",
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
        result["wapi_low"] = forecast_day.get("mintemp_f")
    except Exception as e:
        logger.error("WeatherAPI failed for %s: %s", city_name, e)

    logger.info(
        "WeatherAPI %s: high=%s low=%s",
        city_name,
        result["wapi_high"],
        result["wapi_low"],
    )
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
            result["pirate_low"] = min(today_temps)
    except Exception as e:
        logger.error("Pirate Weather failed for %s: %s", city_name, e)

    logger.info(
        "Pirate Weather %s: high=%s low=%s",
        city_name,
        result["pirate_high"],
        result["pirate_low"],
    )
    return result


# ---------------------------------------------------------------------------
# Combined fetch
# ---------------------------------------------------------------------------

# TOMORROW_CITY_DELAY_SECONDS = 2.5
# TOMORROW_CONCURRENCY = 1
# TOMORROW_MAX_RETRIES = 3


async def _get_with_retry_429(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict,
    timeout: float = 15,
    max_retries: int = 3,
    provider: str = "provider",
    city_name: str = "",
) -> httpx.Response | None:
    for attempt in range(max_retries + 1):
        try:
            resp = await client.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                if attempt == max_retries:
                    logger.warning("%s rate-limited for %s after %s retries", provider, city_name, max_retries)
                    return None
                retry_after = resp.headers.get("Retry-After")
                sleep_s = _retry_sleep_seconds(attempt, retry_after)
                sleep_s = max(sleep_s, TOMORROW_CITY_DELAY_SECONDS)
                logger.warning(
                    "%s rate-limited for %s, retrying in %.2fs (%s/%s)",
                    provider, city_name, sleep_s, attempt + 1, max_retries
                )
                await asyncio.sleep(sleep_s)
                continue

            resp.raise_for_status()
            return resp

        except asyncio.CancelledError:
            logger.warning("%s request cancelled for %s", provider, city_name)
            return None
        except httpx.RequestError as e:
            if attempt == max_retries:
                logger.error("%s request error for %s: %s", provider, city_name, e)
                return None
            sleep_s = max(_retry_sleep_seconds(attempt), TOMORROW_CITY_DELAY_SECONDS)
            logger.warning(
                "%s request error for %s, retrying in %.2fs (%s/%s)",
                provider, city_name, sleep_s, attempt + 1, max_retries
            )
            await asyncio.sleep(sleep_s)
        except httpx.HTTPStatusError as e:
            logger.error("%s failed for %s: %s", provider, city_name, e)
            return None
    return None


async def fetch_city_third_party(
    client: httpx.AsyncClient, city_name: str, city_cfg: dict, tomorrow_sem: asyncio.Semaphore
) -> dict:
    try:
        async with tomorrow_sem:
            tomorrow = await fetch_tomorrow_io(client, city_name, city_cfg)
            await asyncio.sleep(TOMORROW_CITY_DELAY_SECONDS)  # <-- pacing lives HERE

        wapi, pirate = await asyncio.gather(
            fetch_weather_api(client, city_name, city_cfg),
            fetch_pirate_weather(client, city_name, city_cfg),
        )
        return {**tomorrow, **wapi, **pirate}

    except asyncio.CancelledError:
        logger.warning("City task cancelled for %s", city_name)
        return {"city": city_name, "tomorrow_high": None, "tomorrow_low": None,
                "wapi_high": None, "wapi_low": None, "pirate_high": None, "pirate_low": None}


async def fetch_all_third_party() -> list[dict]:
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
    tomorrow_sem = asyncio.Semaphore(TOMORROW_CONCURRENCY)

    async with httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(30.0)) as client:
        tasks = [
            asyncio.create_task(fetch_city_third_party(client, city_name, city_cfg, tomorrow_sem))
            for city_name, city_cfg in CITIES.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    cleaned = []
    for item in results:
        if isinstance(item, Exception):
            logger.error("Third-party task failed: %s", item)
            continue
        cleaned.append(item)
    return cleaned


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = asyncio.run(fetch_all_third_party())
    for r in results:
        print(r)
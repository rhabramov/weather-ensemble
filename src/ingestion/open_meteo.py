"""
Open-Meteo ingestion — no API key required.

Pulls the following forecast members per city:
  Members 5-6:   GFS 2m temperature max/min
  Members 7-8:   NAM 2m temperature max/min
  Members 9-10:  HRRR 2m temperature max/min (derived from hourly)
  Members 11-26: GEFS ensemble members 0-15 (high + low each)
  Members 27-32: ICON-EPS ensemble members 0-5 (high + low each)

Open-Meteo returns temperatures in Celsius by default.
We convert to Fahrenheit throughout.
"""

import logging
import asyncio
from datetime import date, datetime
from zoneinfo import ZoneInfo
from typing import Optional

import httpx

from config.cities import CITIES
from config.settings import GEFS_MEMBERS, ICON_MEMBERS

logger = logging.getLogger(__name__)

BASE_URL      = "https://api.open-meteo.com/v1/forecast"
ENSEMBLE_URL  = "https://ensemble-api.open-meteo.com/v1/ensemble"

def c_to_f(c: Optional[float]) -> Optional[float]:
    if c is None:
        return None
    return round(c * 9 / 5 + 32, 2)


async def fetch_deterministic_models(
    client: httpx.AsyncClient, city_name: str, city_cfg: dict
) -> dict:
    """
    Fetch GFS, NAM, and HRRR daily high/low from Open-Meteo.

    Open-Meteo returns temperature_2m_max and temperature_2m_min as
    daily aggregates when daily= is specified. We pull all three models
    in a single request using the models= parameter.
    """
    params = {
        "latitude":  city_cfg["lat"],
        "longitude": city_cfg["lon"],
        "daily": "temperature_2m_max,temperature_2m_min",
        "models": "gfs_seamless,nam_conus,best_match",  # best_match includes HRRR where available
        "temperature_unit": "fahrenheit",
        "timezone": city_cfg["tz"],
        "forecast_days": 1,
    }

    result = {
        "city": city_name,
        "gfs_high": None, "gfs_low": None,
        "nam_high": None, "nam_low": None,
        "hrrr_high": None, "hrrr_low": None,
    }

    try:
        resp = await client.get(BASE_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        # Open-Meteo returns model-specific keys when multiple models requested
        # Keys look like: temperature_2m_max_gfs_seamless, temperature_2m_max_nam_conus, etc.
        daily = data.get("daily", {})

        def first(vals):
            return vals[0] if vals else None

        result["gfs_high"] = first(daily.get("temperature_2m_max_gfs_seamless", [None]))
        result["gfs_low"]  = first(daily.get("temperature_2m_min_gfs_seamless", [None]))
        result["nam_high"] = first(daily.get("temperature_2m_max_nam_conus", [None]))
        result["nam_low"]  = first(daily.get("temperature_2m_min_nam_conus", [None]))
        result["hrrr_high"] = first(daily.get("temperature_2m_max_best_match", [None]))
        result["hrrr_low"]  = first(daily.get("temperature_2m_min_best_match", [None]))

    except Exception as e:
        logger.error(f"Open-Meteo deterministic fetch failed for {city_name}: {e}")

    logger.info(f"Open-Meteo det {city_name}: GFS={result['gfs_high']}/{result['gfs_low']} "
                f"NAM={result['nam_high']}/{result['nam_low']} HRRR={result['hrrr_high']}/{result['hrrr_low']}")
    return result


async def fetch_gefs_members(
    client: httpx.AsyncClient, city_name: str, city_cfg: dict
) -> dict:
    """
    Fetch GEFS ensemble members 0-15 from Open-Meteo ensemble API.

    Each member is a perturbed-initial-condition forecast from NOAA's
    Global Ensemble Forecast System. 16 members × (high + low) = 32 values.
    """
    member_vars = []
    for m in GEFS_MEMBERS:
        member_vars.append(f"temperature_2m_max_member{m:02d}")
        member_vars.append(f"temperature_2m_min_member{m:02d}")

    params = {
        "latitude":  city_cfg["lat"],
        "longitude": city_cfg["lon"],
        "daily": ",".join(member_vars),
        "models": "gfs_seamless",
        "temperature_unit": "fahrenheit",
        "timezone": city_cfg["tz"],
        "forecast_days": 1,
    }

    result = {"city": city_name}

    try:
        resp = await client.get(ENSEMBLE_URL, params=params, timeout=20)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})

        for m in GEFS_MEMBERS:
            high_key = f"temperature_2m_max_member{m:02d}"
            low_key  = f"temperature_2m_min_member{m:02d}"
            vals_high = daily.get(high_key, [None])
            vals_low  = daily.get(low_key,  [None])
            result[f"gefs_m{m:02d}_high"] = vals_high[0] if vals_high else None
            result[f"gefs_m{m:02d}_low"]  = vals_low[0]  if vals_low  else None

    except Exception as e:
        logger.error(f"GEFS ensemble fetch failed for {city_name}: {e}")
        for m in GEFS_MEMBERS:
            result[f"gefs_m{m:02d}_high"] = None
            result[f"gefs_m{m:02d}_low"]  = None

    logger.info(f"GEFS {city_name}: fetched {len(GEFS_MEMBERS)} members")
    return result


async def fetch_icon_members(
    client: httpx.AsyncClient, city_name: str, city_cfg: dict
) -> dict:
    """
    Fetch ICON-EPS ensemble members 0-5 from Open-Meteo.

    ICON-EPS is DWD Germany's ensemble — independent model, different
    physics parameterizations than GFS, adds genuine diversity.
    """
    member_vars = []
    for m in ICON_MEMBERS:
        member_vars.append(f"temperature_2m_max_member{m:02d}")
        member_vars.append(f"temperature_2m_min_member{m:02d}")

    params = {
        "latitude":  city_cfg["lat"],
        "longitude": city_cfg["lon"],
        "daily": ",".join(member_vars),
        "models": "icon_seamless",
        "temperature_unit": "fahrenheit",
        "timezone": city_cfg["tz"],
        "forecast_days": 1,
    }

    result = {"city": city_name}

    try:
        resp = await client.get(ENSEMBLE_URL, params=params, timeout=20)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})

        for m in ICON_MEMBERS:
            high_key = f"temperature_2m_max_member{m:02d}"
            low_key  = f"temperature_2m_min_member{m:02d}"
            vals_high = daily.get(high_key, [None])
            vals_low  = daily.get(low_key,  [None])
            result[f"icon_m{m:02d}_high"] = vals_high[0] if vals_high else None
            result[f"icon_m{m:02d}_low"]  = vals_low[0]  if vals_low  else None

    except Exception as e:
        logger.error(f"ICON ensemble fetch failed for {city_name}: {e}")
        for m in ICON_MEMBERS:
            result[f"icon_m{m:02d}_high"] = None
            result[f"icon_m{m:02d}_low"]  = None

    logger.info(f"ICON {city_name}: fetched {len(ICON_MEMBERS)} members")
    return result


async def fetch_city_open_meteo(
    client_det: httpx.AsyncClient,
    client_ens: httpx.AsyncClient,
    city_name: str,
    city_cfg: dict,
) -> dict:
    """Fetch all Open-Meteo products for one city and merge into one dict."""
    det, gefs, icon = await asyncio.gather(
        fetch_deterministic_models(client_det, city_name, city_cfg),
        fetch_gefs_members(client_ens, city_name, city_cfg),
        fetch_icon_members(client_ens, city_name, city_cfg),
    )
    return {**det, **gefs, **icon}


async def fetch_all_open_meteo() -> list[dict]:
    """Fetch all Open-Meteo data for all 20 cities concurrently."""
    async with (
        httpx.AsyncClient() as client_det,
        httpx.AsyncClient() as client_ens,
    ):
        tasks = [
            fetch_city_open_meteo(client_det, client_ens, city_name, city_cfg)
            for city_name, city_cfg in CITIES.items()
        ]
        results = await asyncio.gather(*tasks)
    return list(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = asyncio.run(fetch_all_open_meteo())
    for r in results:
        print({k: v for k, v in r.items() if "high" in k or "low" in k or k == "city"})

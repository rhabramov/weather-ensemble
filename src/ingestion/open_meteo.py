"""
Open-Meteo ingestion — no API key required.

Deterministic models (GFS, NAM, HRRR):
  Uses api.open-meteo.com with daily=temperature_2m_max/min directly.
  Members 5-10.

Ensemble members (GEFS 0-15, ICON 0-5):
  Uses ensemble-api.open-meteo.com with hourly=temperature_2m per member,
  then aggregates to daily max/min in local time.
  Members 11-32.
"""

import logging
import asyncio
from datetime import datetime, date
from zoneinfo import ZoneInfo
from typing import Optional

import httpx
import pandas as pd

from config.cities import CITIES
from config.settings import GEFS_MEMBERS, ICON_MEMBERS

logger = logging.getLogger(__name__)

BASE_URL     = "https://api.open-meteo.com/v1/forecast"
ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"


def _daily_max_min_from_hourly(
    hourly_times: list,
    hourly_temps: list,
    local_tz: ZoneInfo,
    today: date,
) -> tuple[Optional[float], Optional[float]]:
    """
    Given parallel lists of ISO time strings and temperatures,
    return (max, min) for hours falling on `today` in local_tz.
    """
    temps_today = []
    for t_str, temp in zip(hourly_times, hourly_temps):
        if temp is None:
            continue
        try:
            dt = datetime.fromisoformat(t_str).replace(tzinfo=ZoneInfo("UTC")).astimezone(local_tz)
            if dt.date() == today:
                temps_today.append(temp)
        except Exception:
            continue
    if not temps_today:
        return None, None
    return max(temps_today), min(temps_today)


async def fetch_deterministic_models(
    client: httpx.AsyncClient, city_name: str, city_cfg: dict
) -> dict:
    """
    Fetch GFS, NAM, HRRR daily high/low from Open-Meteo forecast API.
    These support daily aggregates directly so no hourly→daily step needed.
    """
    params = {
        "latitude":         city_cfg["lat"],
        "longitude":        city_cfg["lon"],
        "daily":            "temperature_2m_max,temperature_2m_min",
        "models":           "gfs_seamless,nam_conus,best_match",
        "temperature_unit": "fahrenheit",
        "timezone":         city_cfg["tz"],
        "forecast_days":    1,
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
        daily = resp.json().get("daily", {})

        def first(key):
            vals = daily.get(key, [None])
            return vals[0] if vals else None

        result["gfs_high"]  = first("temperature_2m_max_gfs_seamless")
        result["gfs_low"]   = first("temperature_2m_min_gfs_seamless")
        result["nam_high"]  = first("temperature_2m_max_nam_conus")
        result["nam_low"]   = first("temperature_2m_min_nam_conus")
        result["hrrr_high"] = first("temperature_2m_max_best_match")
        result["hrrr_low"]  = first("temperature_2m_min_best_match")

    except Exception as e:
        logger.error(f"Open-Meteo deterministic failed for {city_name}: {e}")

    logger.info(
        f"Open-Meteo det {city_name}: "
        f"GFS={result['gfs_high']}/{result['gfs_low']} "
        f"NAM={result['nam_high']}/{result['nam_low']} "
        f"HRRR={result['hrrr_high']}/{result['hrrr_low']}"
    )
    return result


async def fetch_ensemble_members(
    client: httpx.AsyncClient,
    city_name: str,
    city_cfg: dict,
    model: str,
    members: list[int],
    prefix: str,
) -> dict:
    """
    Fetch hourly temperature for each ensemble member and aggregate to daily high/low.

    The ensemble API only supports hourly output — we request all members
    in one call (one variable per member) then aggregate in local time.
    """
    local_tz = ZoneInfo(city_cfg["tz"])
    today    = datetime.now(local_tz).date()

    # Build variable list: temperature_2m_member00, temperature_2m_member01, ...
    member_vars = [f"temperature_2m_member{m:02d}" for m in members]

    params = {
        "latitude":         city_cfg["lat"],
        "longitude":        city_cfg["lon"],
        "hourly":           ",".join(member_vars),
        "models":           model,
        "temperature_unit": "fahrenheit",
        "timezone":         "UTC",   # always UTC from ensemble API; we convert manually
        "forecast_days":    2,       # 2 days to ensure we capture full local day
    }

    result = {"city": city_name}
    for m in members:
        result[f"{prefix}_m{m:02d}_high"] = None
        result[f"{prefix}_m{m:02d}_low"]  = None

    try:
        resp = await client.get(ENSEMBLE_URL, params=params, timeout=30)
        resp.raise_for_status()
        hourly = resp.json().get("hourly", {})
        times  = hourly.get("time", [])

        for m in members:
            var   = f"temperature_2m_member{m:02d}"
            temps = hourly.get(var, [])
            high, low = _daily_max_min_from_hourly(times, temps, local_tz, today)
            result[f"{prefix}_m{m:02d}_high"] = high
            result[f"{prefix}_m{m:02d}_low"]  = low

    except Exception as e:
        logger.error(f"{model} ensemble failed for {city_name}: {e}")

    n_ok = sum(1 for m in members if result[f"{prefix}_m{m:02d}_high"] is not None)
    logger.info(f"{model} {city_name}: {n_ok}/{len(members)} members retrieved")
    return result


async def fetch_city_open_meteo(
    client_det: httpx.AsyncClient,
    client_ens: httpx.AsyncClient,
    city_name: str,
    city_cfg: dict,
) -> dict:
    """Fetch all Open-Meteo products for one city concurrently."""
    det, gefs, icon = await asyncio.gather(
        fetch_deterministic_models(client_det, city_name, city_cfg),
        fetch_ensemble_members(client_ens, city_name, city_cfg,
                               model="gfs_seamless", members=GEFS_MEMBERS, prefix="gefs"),
        fetch_ensemble_members(client_ens, city_name, city_cfg,
                               model="icon_seamless", members=ICON_MEMBERS, prefix="icon"),
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
        print(
            f"{r['city']:20s}  "
            f"GFS={r.get('gfs_high')}/{r.get('gfs_low')}  "
            f"GEFS_m00={r.get('gefs_m00_high')}/{r.get('gefs_m00_low')}  "
            f"ICON_m00={r.get('icon_m00_high')}/{r.get('icon_m00_low')}"
        )
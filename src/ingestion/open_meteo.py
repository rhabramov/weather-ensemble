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


DETERMINISTIC_MODELS = {
    "gfs":  "gfs_seamless",
    "hrrr": "best_match",
}

async def fetch_deterministic_models(
    client: httpx.AsyncClient, city_name: str, city_cfg: dict,
    sem: asyncio.Semaphore,
) -> dict:
    result = {
        "city": city_name,
        "gfs_high": None, "gfs_low": None,
        "nam_high": None, "nam_low": None,
        "hrrr_high": None, "hrrr_low": None,
    }

    async def fetch_one(prefix: str, model: str):
        params = {
            "latitude":         city_cfg["lat"],
            "longitude":        city_cfg["lon"],
            "daily":            "temperature_2m_max,temperature_2m_min",
            "models":           model,
            "temperature_unit": "fahrenheit",
            "timezone":         city_cfg["tz"],
            "forecast_days":    1,
        }
        try:
            async with sem:
                resp = await client.get(BASE_URL, params=params, timeout=20)
            resp.raise_for_status()
            daily = resp.json().get("daily", {})
            result[f"{prefix}_high"] = (daily.get("temperature_2m_max") or [None])[0]
            result[f"{prefix}_low"]  = (daily.get("temperature_2m_min") or [None])[0]
        except Exception as e:
            logger.error("Open-Meteo %s failed for %s: %s", model, city_name, e)

    await asyncio.gather(*[fetch_one(p, m) for p, m in DETERMINISTIC_MODELS.items()])
    logger.info(
        "Open-Meteo det %s: GFS=%s/%s NAM=%s/%s HRRR=%s/%s",
        city_name,
        result["gfs_high"], result["gfs_low"],
        result["nam_high"], result["nam_low"],
        result["hrrr_high"], result["hrrr_low"],
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
    local_tz = ZoneInfo(city_cfg["tz"])
    today    = datetime.now(local_tz).date()

    params = {
        "latitude":         city_cfg["lat"],
        "longitude":        city_cfg["lon"],
        "hourly":           "temperature_2m",   # <-- just this; API returns member00, member01, etc.
        "models":           model,
        "temperature_unit": "fahrenheit",
        "timezone":         "UTC",
        "forecast_days":    2,
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
    det, gefs, icon = await asyncio.gather(
        fetch_deterministic_models(client_det, city_name, city_cfg),
        fetch_ensemble_members(client_ens, city_name, city_cfg,
                               model="gfs_seamless", members=GEFS_MEMBERS, prefix="gefs"),
        fetch_ensemble_members(client_ens, city_name, city_cfg,
                               model="icon_seamless", members=ICON_MEMBERS, prefix="icon"),
    )
    return {**det, **gefs, **icon}


ENSEMBLE_CONCURRENCY = 4  # tune down if still hitting 429
DET_CONCURRENCY = 4  # tune down if still hitting 429

async def fetch_all_open_meteo() -> list[dict]:
    ens_sem = asyncio.Semaphore(ENSEMBLE_CONCURRENCY)
    det_sem = asyncio.Semaphore(DET_CONCURRENCY)

    async def fetch_city(client_det, client_ens, city_name, city_cfg):
        det = await fetch_deterministic_models(client_det, city_name, city_cfg, det_sem)
        
        async def guarded_ensemble(*args, **kwargs):
            async with ens_sem:
                return await fetch_ensemble_members(*args, **kwargs)

        gefs, icon = await asyncio.gather(
            guarded_ensemble(client_ens, city_name, city_cfg,
                             model="gfs_seamless", members=GEFS_MEMBERS, prefix="gefs"),
            guarded_ensemble(client_ens, city_name, city_cfg,
                             model="icon_seamless", members=ICON_MEMBERS, prefix="icon"),
        )
        return {**det, **gefs, **icon}

    async with (
        httpx.AsyncClient() as client_det,
        httpx.AsyncClient() as client_ens,
    ):
        tasks = [
            fetch_city(client_det, client_ens, city_name, city_cfg)
            for city_name, city_cfg in CITIES.items()
        ]
        results = await asyncio.gather(*tasks)
    return list(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = asyncio.run(fetch_all_open_meteo())
    for r in results:
        ens_keys = [k for k in r if 'gefs' in k or 'icon' in k]
        print(f"{r['city']}: ensemble keys = {ens_keys[:4]}")
        print(
            f"{r['city']:20s}  "
            f"GFS={r.get('gfs_high')}/{r.get('gfs_low')}  "
            f"GEFS_m00={r.get('gefs_m00_high')}/{r.get('gefs_m00_low')}  "
            f"ICON_m00={r.get('icon_m00_high')}/{r.get('icon_m00_low')}"
        )
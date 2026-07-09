"""
NWS CLI (Climatological Local Climatological Data) scraper.

Fetches the verified daily high and low temperatures from the NWS CLI
product page for each city. These are the ground truth labels.

CLI reports are typically posted by 6-9 AM local time for the prior day.
The NWS also provides a JSON API endpoint we prefer over HTML scraping.
"""

import re
import logging
import asyncio
from datetime import date, datetime
from typing import Optional

import httpx

from config.cities import CITIES

logger = logging.getLogger(__name__)

# NWS product API — returns the latest CLI text product as JSON
NWS_PRODUCT_URL = "https://api.weather.gov/products/types/CLI/locations/{issuedby}"
NWS_PRODUCT_DETAIL = "https://api.weather.gov/products/{id}"


async def fetch_latest_cli_product_id(
    client: httpx.AsyncClient, issuedby: str
) -> Optional[str]:
    """Get the product ID of the most recent CLI report for a station."""
    url = NWS_PRODUCT_URL.format(issuedby=issuedby)
    try:
        resp = await client.get(url, timeout=15)
        resp.raise_for_status()
        products = resp.json().get("@graph", [])
        if products:
            return products[0]["id"]  # most recent first
    except Exception as e:
        logger.error(f"Failed to fetch CLI product list for {issuedby}: {e}")
    return None


async def fetch_cli_text(client: httpx.AsyncClient, product_id: str) -> Optional[str]:
    """Fetch the raw text of a CLI product by its ID."""
    url = NWS_PRODUCT_DETAIL.format(id=product_id)
    try:
        resp = await client.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json().get("productText", "")
    except Exception as e:
        logger.error(f"Failed to fetch CLI text for product {product_id}: {e}")
    return None


def parse_cli_text(text: str) -> dict:
    """
    Parse NWS CLI product text for daily high, low, and report date.

    CLI format varies slightly by office but the temperature section
    looks like:

    TEMPERATURE (F)                  TODAY   NORMAL  DEPARTURE  LAST YEAR
    MAXIMUM                             72       68          4         69
    MINIMUM                             51       54         -3         52

    Returns dict with keys: report_date, high_temp, low_temp (all may be None)
    """
    result = {"report_date": None, "high_temp": None, "low_temp": None}

    # Extract report date — appears near top as e.g. "CLIMATE REPORT FOR JULY 7 2025"
    date_match = re.search(
        r"CLIMATE\s+(?:REPORT|DATA)\s+FOR\s+([A-Z]+\s+\d+\s+\d{4})", text
    )
    if date_match:
        try:
            result["report_date"] = datetime.strptime(
                date_match.group(1), "%B %d %Y"
            ).date()
        except ValueError:
            pass

    # Extract maximum temperature (daily high)
    max_match = re.search(
        r"MAXIMUM\s+(\d+)\s+\d+\s+[+-]?\d+", text
    )
    if max_match:
        result["high_temp"] = int(max_match.group(1))

    # Extract minimum temperature (daily low)
    min_match = re.search(
        r"MINIMUM\s+(\d+)\s+\d+\s+[+-]?\d+", text
    )
    if min_match:
        result["low_temp"] = int(min_match.group(1))

    return result


async def fetch_city_cli(
    client: httpx.AsyncClient, city_name: str, city_cfg: dict
) -> dict:
    """Fetch and parse the CLI report for one city."""
    issuedby = city_cfg["cli_issuedby"]
    product_id = await fetch_latest_cli_product_id(client, issuedby)
    if not product_id:
        return {"city": city_name, "report_date": None, "high_temp": None, "low_temp": None}

    text = await fetch_cli_text(client, product_id)
    if not text:
        return {"city": city_name, "report_date": None, "high_temp": None, "low_temp": None}

    parsed = parse_cli_text(text)
    parsed["city"] = city_name
    logger.info(
        f"CLI {city_name}: date={parsed['report_date']} "
        f"high={parsed['high_temp']} low={parsed['low_temp']}"
    )
    return parsed


async def fetch_all_cli_reports() -> list[dict]:
    """
    Fetch CLI reports for all 20 cities concurrently.
    Returns list of dicts with city, report_date, high_temp, low_temp.
    """
    headers = {"User-Agent": "WeatherEnsemble/1.0 (contact@yourdomain.com)"}
    async with httpx.AsyncClient(headers=headers) as client:
        tasks = [
            fetch_city_cli(client, city_name, city_cfg)
            for city_name, city_cfg in CITIES.items()
        ]
        results = await asyncio.gather(*tasks)
    return list(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = asyncio.run(fetch_all_cli_reports())
    for r in results:
        print(r)

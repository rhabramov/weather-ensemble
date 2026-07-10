"""
Fetch missing san_antonio climatological normals and append to climo_normals.csv.
Run: python scripts/fix_san_antonio_climo.py
"""
import httpx
import pandas as pd
from config.settings import PROCESSED_DIR

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude":         29.5341,
    "longitude":        -98.4698,
    "daily":            "temperature_2m_max,temperature_2m_min",
    "temperature_unit": "fahrenheit",
    "timezone":         "America/Chicago",
    "start_date":       "1991-01-01",
    "end_date":         "2020-12-31",
}

print("Fetching san_antonio climo (1991-2020)...")
r = httpx.get(url, params=params, timeout=60)
print(f"Status: {r.status_code}")
r.raise_for_status()

daily = r.json().get("daily", {})
df = pd.DataFrame({
    "date": pd.to_datetime(daily.get("time", [])),
    "high": daily.get("temperature_2m_max", []),
    "low":  daily.get("temperature_2m_min", []),
})
df["doy"] = df["date"].dt.dayofyear
normals = df.groupby("doy")[["high", "low"]].mean().reset_index()
normals.columns = ["doy", "climo_high", "climo_low"]
normals["city"] = "san_antonio"
normals = normals[["city", "doy", "climo_high", "climo_low"]]

path = PROCESSED_DIR / "climo_normals.csv"
existing = pd.read_csv(path)

# Remove any partial san_antonio rows if present
existing = existing[existing["city"] != "san_antonio"]
combined = pd.concat([existing, normals], ignore_index=True)
combined = combined.sort_values(["city", "doy"]).reset_index(drop=True)
combined.to_csv(path, index=False)

print(f"Done. Added {len(normals)} rows.")
print(f"Total: {len(combined)} rows across {combined['city'].nunique()} cities.")
print(f"San Antonio mean climo: high={normals['climo_high'].mean():.1f}F  low={normals['climo_low'].mean():.1f}F")
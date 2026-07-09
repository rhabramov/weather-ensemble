"""
Settings and API key management. All secrets live in .env.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# API keys
TOMORROW_IO_KEY   = os.getenv("TOMORROW_IO_KEY", "")
WEATHER_API_KEY   = os.getenv("WEATHER_API_KEY", "")
PIRATE_WEATHER_KEY = os.getenv("PIRATE_WEATHER_KEY", "")
NOAA_CDO_KEY      = os.getenv("NOAA_CDO_KEY", "")

# Paths
BASE_DIR       = Path(__file__).parent.parent
DATA_DIR       = BASE_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
MODELS_DIR     = DATA_DIR / "models"
LOGS_DIR       = BASE_DIR / "logs"

for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Postgres connection string (optional — falls back to SQLite if not set)
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/weather.db")

# Scheduler timezone
SCHEDULER_TZ = "America/New_York"

# Open-Meteo ensemble members to pull (0–30 available for GEFS)
GEFS_MEMBERS = list(range(16))   # members 0–15
ICON_MEMBERS = list(range(6))    # members 0–5

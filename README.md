# Weather Ensemble — Setup & Run Guide

## What this is

XGBoost ensemble model that predicts daily high and low temperatures for 20 US cities.
Runs at 9 AM ET (primary) and every hour on the hour (updates).
Ground truth: NWS CLI (Climatological Local Climatological Data) verified temps.
34 forecast members sourced from NWS, Open-Meteo (GFS/NAM/HRRR + GEFS/ICON ensembles), and 3 third-party APIs.

---

## Prerequisites

- Python 3.11+
- Linux server (EC2 t3.medium, DigitalOcean Droplet, etc.) or local Mac/Linux
- 4 free API keys (5 minutes to get all four)

---

## Step 1 — Get API Keys

| Key | Where | Time |
|-----|-------|------|
| NOAA CDO | https://www.ncdc.noaa.gov/cdo-web/token | Instant (email) |
| Tomorrow.io | https://app.tomorrow.io/development/keys | 2 min |
| WeatherAPI | https://www.weatherapi.com/signup.aspx | 2 min |
| Pirate Weather | https://pirateweather.net/en/latest/api/ | 2 min |

---

## Step 2 — Install

```bash
git clone <your-repo>
cd weather_ensemble

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.template .env
# Edit .env and paste your 4 keys
```

---

## Step 3 — Build Training Data (one-time, ~1-2 hours)

```bash
# Pull 10 years of verified highs/lows from NOAA CDO
python -m src.ingestion.historical_labels

# Fetch ERA5 reanalysis features and join to actuals
python -m src.models.build_training_data
```

This produces `data/processed/training_data.csv` (~750K rows: 20 cities × 365 days × 10 years).

---

## Step 4 — Train Models

```bash
python -m src.models.xgb_model data/processed/training_data.csv
```

Trains two XGBoost models (high + low), runs 5-fold time-series CV, prints per-city MAE.
Saves models to `data/models/`.

Expect initial MAE of ~2-4°F on historical ERA5 features.
MAE will drop significantly once 6+ months of live forecasts accumulate (real ensemble spread is a strong signal).

---

## Step 5 — Run the Scheduler

```bash
python -m src.scheduler.scheduler
```

Runs indefinitely. Logs to `logs/scheduler.log` and console.

**To keep it running after you close your SSH session:**
```bash
# Option A: nohup (simple)
nohup python -m src.scheduler.scheduler > logs/scheduler.log 2>&1 &

# Option B: systemd service (recommended for production)
# See systemd/weather-ensemble.service in this repo
```

---

## Step 6 — Check Predictions

```python
from src.models.database import get_recent_predictions
df = get_recent_predictions(days=7)
print(df[["city", "forecast_date", "run_type", "pred_high", "pred_low",
          "actual_high", "actual_low", "error_high", "error_low"]])
```

---

## Systemd Service (production)

```ini
# /etc/systemd/system/weather-ensemble.service
[Unit]
Description=Weather Ensemble Forecast System
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/weather_ensemble
ExecStart=/home/ubuntu/weather_ensemble/venv/bin/python -m src.scheduler.scheduler
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable weather-ensemble
sudo systemctl start weather-ensemble
sudo systemctl status weather-ensemble
```

---

## Monthly Retrain

Once you have 3+ months of live predictions saved in the DB, export them
and retrain. Live data has real ensemble spread (vs. ERA5 proxy), so model
performance improves meaningfully.

```bash
# Export live training data from DB
python -c "
from src.models.database import get_recent_predictions
df = get_recent_predictions(days=365)
df.to_csv('data/processed/live_training_data.csv', index=False)
"

# Retrain
python -m src.models.xgb_model data/processed/live_training_data.csv
```

---

## Project Structure

```
weather_ensemble/
├── config/
│   ├── cities.py          # All 20 cities: coords, NWS offices, CLI codes
│   └── settings.py        # API keys, paths, constants
├── src/
│   ├── ingestion/
│   │   ├── cli_scraper.py        # NWS CLI verified actuals (ground truth)
│   │   ├── nws_forecast.py       # NWS daily + hourly forecast (members 1-4)
│   │   ├── open_meteo.py         # GFS/NAM/HRRR + GEFS/ICON members (5-32)
│   │   ├── third_party.py        # Tomorrow.io, WeatherAPI, Pirate Weather (33-35)
│   │   └── historical_labels.py  # NOAA CDO bulk historical pull (one-time)
│   ├── features/
│   │   └── feature_builder.py    # Merges all sources → 20-row feature matrix
│   ├── models/
│   │   ├── xgb_model.py          # Train, evaluate, predict
│   │   ├── database.py           # SQLAlchemy ORM: predictions, actuals, run_log
│   │   └── build_training_data.py # One-time: join ERA5 history to actuals
│   └── scheduler/
│       └── scheduler.py          # APScheduler: 9 AM + hourly jobs
├── data/
│   ├── raw/                      # Raw API responses (optional cache)
│   ├── processed/                # CSVs, training data
│   └── models/                   # Trained model pickles
├── logs/
│   └── scheduler.log
├── .env.template
├── requirements.txt
└── README.md
```

---

## Forecast Members Reference

| # | Member | Source | Key |
|---|--------|--------|-----|
| 1-2 | NWS Official high/low | api.weather.gov | Free |
| 3-4 | NWS Hourly-derived high/low | api.weather.gov | Free |
| 5-6 | GFS high/low | Open-Meteo | Free |
| 7-8 | NAM high/low | Open-Meteo | Free |
| 9-10 | HRRR high/low | Open-Meteo | Free |
| 11-26 | GEFS members 0-15 | Open-Meteo ensemble API | Free |
| 27-32 | ICON-EPS members 0-5 | Open-Meteo ensemble API | Free |
| 33 | Tomorrow.io | tomorrow.io | Free tier |
| 34 | WeatherAPI | weatherapi.com | Free tier |
| 35 | Pirate Weather | pirateweather.net | Free tier |

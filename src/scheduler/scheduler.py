"""
APScheduler-based job scheduler.

Jobs:
  1. morning_run()  — 9:00 AM ET daily
     - Scrapes overnight CLI reports (yesterday's actuals)
     - Pulls all forecast sources
     - Runs model inference
     - Saves predictions + actuals to DB

  2. hourly_update() — every hour on the hour
     - Pulls all forecast sources (no CLI scrape needed)
     - Runs model inference with updated forecasts
     - Saves updated predictions to DB

Run with: python -m src.scheduler.scheduler
Runs indefinitely as a daemon process.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, date

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import SCHEDULER_TZ, LOGS_DIR
from src.features.feature_builder import build_feature_matrix
from src.ingestion.cli_scraper import fetch_all_cli_reports
from src.models.database import (
    create_tables, save_predictions, save_actuals, log_run
)
from src.models.xgb_model import load_models, predict

# Logging setup — file + console
log_path = LOGS_DIR / "scheduler.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


def run_morning():
    """
    9 AM ET job.
    1. Fetch CLI actuals for yesterday → save to DB
    2. Build feature matrix
    3. Run inference
    4. Save predictions
    """
    logger.info("=" * 60)
    logger.info("MORNING RUN starting")
    start = time.time()
    status = "success"
    notes = ""

    try:
        # Step 1: CLI actuals (yesterday's verified high/low)
        logger.info("Fetching CLI reports...")
        cli_reports = asyncio.run(fetch_all_cli_reports())
        # Rename keys to match DB schema
        for r in cli_reports:
            r["actual_high"] = r.pop("high_temp", None)
            r["actual_low"]  = r.pop("low_temp", None)
        save_actuals(cli_reports)

        # Step 2: Feature matrix
        logger.info("Building feature matrix...")
        feature_df = asyncio.run(build_feature_matrix(target_date=date.today()))

        # Step 3: Inference
        logger.info("Running model inference...")
        model_high, model_low, city_encoder = load_models()
        preds_df = predict(feature_df, model_high, model_low, city_encoder)

        # Step 4: Save
        save_predictions(preds_df, run_type="9am")

        n_cities = len(preds_df)
        logger.info(f"Morning run complete: {n_cities} cities predicted")
        for _, row in preds_df.iterrows():
            logger.info(f"  {row['city']:20s}  High={row['pred_high']}°F  Low={row['pred_low']}°F")

    except Exception as e:
        status = "failed"
        notes = str(e)
        logger.exception(f"Morning run failed: {e}")
        n_cities = 0

    duration = time.time() - start
    log_run("9am", status, n_cities if status == "success" else 0, duration, notes)
    logger.info(f"Morning run finished in {duration:.1f}s with status={status}")


def run_hourly():
    """
    Hourly job (every hour on the hour, except 9 AM which is handled above).
    Pulls refreshed forecasts and updates predictions.
    """
    now = datetime.now(timezone.utc)
    logger.info(f"HOURLY UPDATE starting at {now.isoformat()}")
    start = time.time()
    status = "success"
    notes = ""
    n_cities = 0

    try:
        feature_df = asyncio.run(build_feature_matrix(target_date=date.today()))
        model_high, model_low, city_encoder = load_models()
        preds_df = predict(feature_df, model_high, model_low, city_encoder)
        save_predictions(preds_df, run_type="hourly")
        n_cities = len(preds_df)
        logger.info(f"Hourly update complete: {n_cities} cities updated")

    except Exception as e:
        status = "failed"
        notes = str(e)
        logger.exception(f"Hourly update failed: {e}")

    duration = time.time() - start
    log_run("hourly", status, n_cities, duration, notes)


def main():
    logger.info("Initializing database...")
    create_tables()

    logger.info("Loading models...")
    try:
        load_models()
        logger.info("Models loaded successfully")
    except FileNotFoundError:
        logger.warning(
            "No trained models found. Train first with: "
            "python -m src.models.xgb_model <training_data.csv>"
        )

    scheduler = BlockingScheduler(timezone=SCHEDULER_TZ)

    # 9 AM ET every day
    scheduler.add_job(
        run_morning,
        CronTrigger(hour=9, minute=0, timezone=SCHEDULER_TZ),
        id="morning_run",
        name="9 AM Morning Run",
        max_instances=1,
        coalesce=True,
    )

    # Every hour on the hour (APScheduler skips if morning job is already running)
    scheduler.add_job(
        run_hourly,
        CronTrigger(minute=0, timezone=SCHEDULER_TZ),
        id="hourly_update",
        name="Hourly Forecast Update",
        max_instances=1,
        coalesce=True,
    )

    logger.info("Scheduler started. Jobs:")
    for job in scheduler.get_jobs():
        logger.info(f"  {job.name}: {job.trigger}")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
        scheduler.shutdown()


if __name__ == "__main__":
    main()

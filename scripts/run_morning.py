"""
Morning run — 9 AM ET.
1. Fetch CLI actuals (yesterday's verified high/low)
2. Build feature matrix
3. Run inference
4. Save predictions + actuals to CSV
5. Send email with today's forecast + yesterday's accuracy
"""

import asyncio
import logging
import sys
import time
from datetime import date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


async def main():
    start = time.time()
    logger.info("=== MORNING RUN starting ===")

    from src.ingestion.cli_scraper import fetch_all_cli_reports
    from src.features.feature_builder import build_feature_matrix
    from src.models.csv_store import save_predictions, save_actuals, log_run, get_yesterday_accuracy
    from src.models.xgb_model import load_models, predict
    from src.notifications.email_digest import send_forecast_email

    # Step 1: CLI actuals for yesterday
    logger.info("Fetching CLI reports...")
    cli_reports = await fetch_all_cli_reports()
    for r in cli_reports:
        r["actual_high"] = r.pop("high_temp", None)
        r["actual_low"]  = r.pop("low_temp", None)
    save_actuals(cli_reports)

    # Step 2: Feature matrix
    logger.info("Building feature matrix...")
    feature_df = await build_feature_matrix(target_date=date.today())

    # Step 3: Inference
    logger.info("Running inference...")
    model_high, model_low, city_encoder = load_models()
    preds_df = predict(feature_df, model_high, model_low, city_encoder)
    save_predictions(preds_df, run_type="morning")

    # Step 4: Email
    logger.info("Sending email...")
    spread_cols = ["city", "ens_high_spread"]
    available = [c for c in spread_cols if c in feature_df.columns]
    preds_with_spread = (
        preds_df.merge(feature_df[available], on="city", how="left")
        if len(available) == 2 else preds_df
    )
    yesterday_accuracy = get_yesterday_accuracy()
    send_forecast_email(
        predictions_today=preds_with_spread,
        predictions_yesterday=yesterday_accuracy,
        forecast_date=date.today(),
        subject_prefix="🌡️ Morning Forecast —",
    )

    duration = time.time() - start
    log_run("morning", "success", len(preds_df), duration)
    logger.info(f"Morning run complete in {duration:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())

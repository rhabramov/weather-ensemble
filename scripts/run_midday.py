"""
Midday run — Noon ET.
Refreshes forecasts with updated model runs (NWS, GFS, etc. update
several times per day). Sends a noon update email.
No CLI scrape — actuals aren't posted until next morning.
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
    logger.info("=== MIDDAY UPDATE starting ===")

    from src.features.feature_builder import build_feature_matrix
    from src.models.csv_store import save_predictions, log_run
    from src.models.xgb_model import load_models, predict
    from src.notifications.email_digest import send_forecast_email

    logger.info("Building feature matrix with refreshed forecasts...")
    feature_df = await build_feature_matrix(target_date=date.today())

    logger.info("Running inference...")
    model_high, model_low, city_encoder, corrector = load_models()
    preds_df = predict(feature_df, model_high, model_low, city_encoder, corrector)
    save_predictions(preds_df, run_type="midday")

    logger.info("Sending noon update email...")
    spread_cols = ["city", "ens_high_spread"]
    available = [c for c in spread_cols if c in feature_df.columns]
    preds_with_spread = (
        preds_df.merge(feature_df[available], on="city", how="left")
        if len(available) == 2 else preds_df
    )
    send_forecast_email(
        predictions_today=preds_with_spread,
        predictions_yesterday=None,
        forecast_date=date.today(),
        subject_prefix="🌡️ Noon Update —",
    )

    duration = time.time() - start
    log_run("midday", "success", len(preds_df), duration)
    logger.info(f"Midday update complete in {duration:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())

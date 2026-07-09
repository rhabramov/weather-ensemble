"""
Daily forecast email digest.

Sends a formatted HTML email every morning with:
  - Today's predicted high/low for all 20 cities
  - Yesterday's prediction vs actual (error)
  - Any cities where model confidence is low (high ensemble spread)

Uses Gmail SMTP with an App Password (not your regular Gmail password).
Setup: Google Account → Security → 2FA enabled → App Passwords → generate one.

Required .env additions:
  EMAIL_FROM=you@gmail.com
  EMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx   (16-char Gmail App Password)
  EMAIL_TO=you@gmail.com                   (can be same address)
"""

import logging
import os
import smtplib
from datetime import date, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

EMAIL_FROM         = os.getenv("EMAIL_FROM", "")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "")
EMAIL_TO           = os.getenv("EMAIL_TO", "")

CITY_DISPLAY = {
    "seattle":        "Seattle",
    "los_angeles":    "Los Angeles",
    "miami":          "Miami",
    "new_york":       "New York",
    "minneapolis":    "Minneapolis",
    "houston":        "Houston",
    "denver":         "Denver",
    "boston":         "Boston",
    "chicago":        "Chicago",
    "dallas":         "Dallas",
    "philadelphia":   "Philadelphia",
    "san_francisco":  "San Francisco",
    "las_vegas":      "Las Vegas",
    "oklahoma_city":  "Oklahoma City",
    "austin":         "Austin",
    "san_antonio":    "San Antonio",
    "phoenix":        "Phoenix",
    "new_orleans":    "New Orleans",
    "atlanta":        "Atlanta",
    "washington_dc":  "Washington DC",
}


def format_error(err: Optional[float]) -> str:
    if err is None or pd.isna(err):
        return "—"
    sign = "+" if err > 0 else ""
    return f"{sign}{err:.1f}°F"


def format_temp(t: Optional[float]) -> str:
    if t is None or pd.isna(t):
        return "—"
    return f"{t:.0f}°F"


def build_html(
    predictions_today: pd.DataFrame,
    predictions_yesterday: Optional[pd.DataFrame],
    forecast_date: date,
) -> str:
    """Build the HTML email body."""

    # Sort cities alphabetically for display
    predictions_today = predictions_today.sort_values("city")

    rows_today = ""
    for _, row in predictions_today.iterrows():
        city = CITY_DISPLAY.get(row["city"], row["city"])
        high = format_temp(row.get("pred_high"))
        low  = format_temp(row.get("pred_low"))

        # Ensemble spread as confidence indicator
        spread = row.get("ens_high_spread")
        if spread is not None and not pd.isna(spread):
            if spread > 10:
                confidence = "🔴 Low"
            elif spread > 5:
                confidence = "🟡 Med"
            else:
                confidence = "🟢 High"
        else:
            confidence = "—"

        rows_today += f"""
        <tr>
            <td style="padding:8px 12px;border-bottom:1px solid #eee;">{city}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #eee;text-align:center;font-weight:bold;color:#c0392b;">{high}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #eee;text-align:center;font-weight:bold;color:#2980b9;">{low}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #eee;text-align:center;font-size:12px;">{confidence}</td>
        </tr>"""

    # Yesterday's accuracy table
    accuracy_section = ""
    if predictions_yesterday is not None and not predictions_yesterday.empty:
        has_actuals = predictions_yesterday[["actual_high", "actual_low"]].notna().any().any()
        if has_actuals:
            rows_yesterday = ""
            maes = []
            for _, row in predictions_yesterday.sort_values("city").iterrows():
                city = CITY_DISPLAY.get(row["city"], row["city"])
                pred_h = format_temp(row.get("pred_high"))
                pred_l = format_temp(row.get("pred_low"))
                act_h  = format_temp(row.get("actual_high"))
                act_l  = format_temp(row.get("actual_low"))
                err_h  = format_error(row.get("error_high"))
                err_l  = format_error(row.get("error_low"))

                # Color the error
                def err_color(e):
                    if e is None or pd.isna(e):
                        return "#666"
                    return "#c0392b" if abs(e) > 5 else "#27ae60" if abs(e) <= 2 else "#e67e22"

                eh = row.get("error_high")
                el = row.get("error_low")
                color_h = err_color(eh)
                color_l = err_color(el)

                if eh is not None and not pd.isna(eh):
                    maes.append(abs(eh))
                if el is not None and not pd.isna(el):
                    maes.append(abs(el))

                rows_yesterday += f"""
                <tr>
                    <td style="padding:6px 12px;border-bottom:1px solid #eee;">{city}</td>
                    <td style="padding:6px 12px;border-bottom:1px solid #eee;text-align:center;">{pred_h} → {act_h}</td>
                    <td style="padding:6px 12px;border-bottom:1px solid #eee;text-align:center;color:{color_h};">{err_h}</td>
                    <td style="padding:6px 12px;border-bottom:1px solid #eee;text-align:center;">{pred_l} → {act_l}</td>
                    <td style="padding:6px 12px;border-bottom:1px solid #eee;text-align:center;color:{color_l};">{err_l}</td>
                </tr>"""

            overall_mae = f"{sum(maes)/len(maes):.2f}°F" if maes else "—"
            yesterday_str = (forecast_date - timedelta(days=1)).strftime("%B %d")

            accuracy_section = f"""
            <h2 style="color:#2c3e50;margin-top:32px;">Yesterday's Accuracy ({yesterday_str}) &nbsp;
                <span style="font-size:14px;font-weight:normal;color:#7f8c8d;">Overall MAE: {overall_mae}</span>
            </h2>
            <table style="border-collapse:collapse;width:100%;font-family:Arial,sans-serif;font-size:13px;">
                <thead>
                    <tr style="background:#f8f9fa;">
                        <th style="padding:8px 12px;text-align:left;">City</th>
                        <th style="padding:8px 12px;text-align:center;">High (Pred→Actual)</th>
                        <th style="padding:8px 12px;text-align:center;">Error</th>
                        <th style="padding:8px 12px;text-align:center;">Low (Pred→Actual)</th>
                        <th style="padding:8px 12px;text-align:center;">Error</th>
                    </tr>
                </thead>
                <tbody>{rows_yesterday}</tbody>
            </table>"""

    date_str = forecast_date.strftime("%A, %B %d, %Y")

    html = f"""
    <html><body style="font-family:Arial,sans-serif;max-width:700px;margin:0 auto;color:#2c3e50;">
        <div style="background:#2c3e50;padding:20px 24px;border-radius:8px 8px 0 0;">
            <h1 style="color:white;margin:0;font-size:20px;">🌡️ Weather Ensemble Forecast</h1>
            <p style="color:#bdc3c7;margin:4px 0 0;">{date_str}</p>
        </div>
        <div style="padding:24px;border:1px solid #eee;border-top:none;border-radius:0 0 8px 8px;">

            <h2 style="color:#2c3e50;">Today's Predictions</h2>
            <table style="border-collapse:collapse;width:100%;font-family:Arial,sans-serif;font-size:13px;">
                <thead>
                    <tr style="background:#f8f9fa;">
                        <th style="padding:8px 12px;text-align:left;">City</th>
                        <th style="padding:8px 12px;text-align:center;">High</th>
                        <th style="padding:8px 12px;text-align:center;">Low</th>
                        <th style="padding:8px 12px;text-align:center;">Confidence</th>
                    </tr>
                </thead>
                <tbody>{rows_today}</tbody>
            </table>
            <p style="font-size:11px;color:#95a5a6;margin-top:8px;">
                Confidence based on ensemble spread across 34 forecast members.
                🟢 &lt;5°F spread &nbsp; 🟡 5–10°F &nbsp; 🔴 &gt;10°F
            </p>

            {accuracy_section}

            <p style="font-size:11px;color:#bdc3c7;margin-top:32px;border-top:1px solid #eee;padding-top:12px;">
                Weather Ensemble | XGBoost model trained on NWS CLI verified temps<br>
                Sources: NWS, GFS, NAM, HRRR, GEFS (16 members), ICON-EPS (6 members), Tomorrow.io, WeatherAPI, Pirate Weather
            </p>
        </div>
    </body></html>
    """
    return html


def send_forecast_email(
    predictions_today: pd.DataFrame,
    predictions_yesterday: Optional[pd.DataFrame] = None,
    forecast_date: Optional[date] = None,
    subject_prefix: str = "🌡️ Morning Forecast —",
) -> bool:
    """
    Send the daily forecast digest email.
    Returns True on success, False on failure.
    """
    if not all([EMAIL_FROM, EMAIL_APP_PASSWORD, EMAIL_TO]):
        logger.warning("Email credentials not set — skipping email send")
        return False

    if forecast_date is None:
        forecast_date = date.today()

    date_str = forecast_date.strftime("%A %b %d")
    subject = f"{subject_prefix} {date_str}"

    html_body = build_html(predictions_today, predictions_yesterday, forecast_date)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = EMAIL_FROM
    msg["To"]      = EMAIL_TO
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        logger.info(f"Forecast email sent to {EMAIL_TO}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False


if __name__ == "__main__":
    # Quick test with dummy data
    import numpy as np
    logging.basicConfig(level=logging.INFO)

    dummy = pd.DataFrame({
        "city": ["seattle", "new_york", "miami", "chicago", "phoenix"],
        "pred_high": [72.0, 85.0, 91.0, 78.0, 108.0],
        "pred_low":  [58.0, 70.0, 79.0, 63.0, 88.0],
        "ens_high_spread": [3.2, 7.1, 2.8, 11.4, 4.0],
        "forecast_date": [date.today()] * 5,
    })
    send_forecast_email(dummy)

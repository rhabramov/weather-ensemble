"""
Database layer — stores predictions and actuals.

Uses SQLAlchemy with SQLite by default (zero-config) or Postgres if
DATABASE_URL is set in .env. Schema is created on first run.

Tables:
  predictions  — one row per city per run (timestamped)
  actuals      — one row per city per date (from CLI scraper)
  run_log      — one row per scheduler run (for monitoring)
"""

import logging
from datetime import datetime, date, timezone

from sqlalchemy import (
    create_engine, Column, String, Float, Integer,
    DateTime, Date, Text, UniqueConstraint, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from config.settings import DATABASE_URL

logger = logging.getLogger(__name__)

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    city          = Column(String(50), nullable=False)
    forecast_date = Column(Date, nullable=False)
    run_time      = Column(DateTime(timezone=True), nullable=False)
    run_type      = Column(String(10), nullable=False)  # "9am" | "hourly"
    pred_high     = Column(Float)
    pred_low      = Column(Float)

    __table_args__ = (
        Index("ix_pred_city_date", "city", "forecast_date"),
    )


class Actual(Base):
    __tablename__ = "actuals"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    city        = Column(String(50), nullable=False)
    report_date = Column(Date, nullable=False)
    actual_high = Column(Float)
    actual_low  = Column(Float)
    ingested_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("city", "report_date", name="uq_actual_city_date"),
    )


class RunLog(Base):
    __tablename__ = "run_log"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    run_time   = Column(DateTime(timezone=True), nullable=False)
    run_type   = Column(String(10), nullable=False)
    status     = Column(String(20), nullable=False)  # "success" | "partial" | "failed"
    n_cities   = Column(Integer)
    duration_s = Column(Float)
    notes      = Column(Text)


def create_tables():
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified")


def save_predictions(preds_df, run_type: str = "hourly"):
    """Save a predictions DataFrame to the database."""
    with SessionLocal() as session:
        for _, row in preds_df.iterrows():
            pred = Prediction(
                city=row["city"],
                forecast_date=row["forecast_date"],
                run_time=datetime.now(timezone.utc),
                run_type=run_type,
                pred_high=row.get("pred_high"),
                pred_low=row.get("pred_low"),
            )
            session.add(pred)
        session.commit()
    logger.info(f"Saved {len(preds_df)} predictions (run_type={run_type})")


def save_actuals(cli_reports: list[dict]):
    """
    Upsert CLI actuals into the actuals table.
    Skips cities with missing report_date or both temps null.
    """
    saved, skipped = 0, 0
    with SessionLocal() as session:
        for r in cli_reports:
            if not r.get("report_date") or (r.get("actual_high") is None and r.get("actual_low") is None):
                skipped += 1
                continue
            existing = session.query(Actual).filter_by(
                city=r["city"], report_date=r["report_date"]
            ).first()
            if existing:
                existing.actual_high = r.get("actual_high") or existing.actual_high
                existing.actual_low  = r.get("actual_low")  or existing.actual_low
            else:
                session.add(Actual(
                    city=r["city"],
                    report_date=r["report_date"],
                    actual_high=r.get("actual_high"),
                    actual_low=r.get("actual_low"),
                ))
            saved += 1
        session.commit()
    logger.info(f"Actuals: {saved} saved, {skipped} skipped")


def log_run(run_type: str, status: str, n_cities: int, duration_s: float, notes: str = ""):
    with SessionLocal() as session:
        session.add(RunLog(
            run_time=datetime.now(timezone.utc),
            run_type=run_type,
            status=status,
            n_cities=n_cities,
            duration_s=duration_s,
            notes=notes,
        ))
        session.commit()


def get_recent_predictions(days: int = 7) -> "pd.DataFrame":
    """Fetch predictions from the last N days for review."""
    import pandas as pd
    from sqlalchemy import text
    with SessionLocal() as session:
        result = session.execute(
            text("""
                SELECT p.city, p.forecast_date, p.run_time, p.run_type,
                       p.pred_high, p.pred_low,
                       a.actual_high, a.actual_low,
                       (p.pred_high - a.actual_high) AS error_high,
                       (p.pred_low  - a.actual_low)  AS error_low
                FROM predictions p
                LEFT JOIN actuals a
                  ON p.city = a.city AND p.forecast_date = a.report_date
                WHERE p.forecast_date >= date('now', :offset)
                ORDER BY p.forecast_date DESC, p.city, p.run_time DESC
            """),
            {"offset": f"-{days} days"}
        )
        return pd.DataFrame(result.fetchall(), columns=result.keys())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_tables()
    print("Database initialized at:", DATABASE_URL)

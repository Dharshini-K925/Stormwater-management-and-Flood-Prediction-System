"""Database layer for local SQLite development and hosted PostgreSQL deployment."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.types import JSON

ROOT = Path(__file__).resolve().parent
DEFAULT_SQLITE = f"sqlite:///{(ROOT / 'data' / 'flood_explorer.db').as_posix()}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_SQLITE).strip()
# Railway and some hosts provide postgres://; SQLAlchemy 2 uses the psycopg driver explicitly.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql+psycopg://" + DATABASE_URL[len("postgres://"):]
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = "postgresql+psycopg://" + DATABASE_URL[len("postgresql://"):]

engine_options = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite:"):
    engine_options["connect_args"] = {"check_same_thread": False}
engine = create_engine(DATABASE_URL, **engine_options)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)

class Base(DeclarativeBase):
    pass

class TrainingSample(Base):
    __tablename__ = "training_samples"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_id: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    target: Mapped[int] = mapped_column(Integer, index=True)
    features_json: Mapped[dict] = mapped_column(JSON)
    source: Mapped[str] = mapped_column(String(40), default="synthetic_csv")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class RainfallReading(Base):
    __tablename__ = "rainfall_readings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    rainfall_1h_mm: Mapped[float] = mapped_column(Float)
    rainfall_24h_mm: Mapped[float] = mapped_column(Float)
    antecedent_3d_mm: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(30), default="website")
    device_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)

class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    rainfall_1h_mm: Mapped[float] = mapped_column(Float)
    rainfall_24h_mm: Mapped[float] = mapped_column(Float)
    antecedent_3d_mm: Mapped[float] = mapped_column(Float)
    probability_pct: Mapped[float] = mapped_column(Float)
    risk: Mapped[str] = mapped_column(String(20), index=True)
    rainfall_reading_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    details_json: Mapped[dict] = mapped_column(JSON)

class VerifiedFloodObservation(Base):
    """Separate verified outcomes from model predictions; never treat a prediction as ground truth."""
    __tablename__ = "verified_flood_observations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    flood_occurred: Mapped[bool] = mapped_column(Boolean)
    water_depth_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(200))
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

def init_database(csv_path: Path) -> None:
    """Create tables and import the included synthetic CSV once if the training table is empty."""
    Base.metadata.create_all(engine)
    with SessionLocal() as session:
        existing = session.scalar(select(func.count()).select_from(TrainingSample)) or 0
        if existing == 0:
            frame = pd.read_csv(csv_path)
            rows = []
            for _, row in frame.iterrows():
                values = row.to_dict()
                sample_id = str(values.get("sample_id", f"synthetic-{len(rows) + 1}"))
                target = int(values.pop("flood_occurrence_synthetic"))
                lat = float(values["latitude"])
                lon = float(values["longitude"])
                # Convert NumPy scalars to regular Python values for JSON serialization.
                features = {str(k): (v.item() if hasattr(v, "item") else v) for k, v in values.items()}
                rows.append(TrainingSample(sample_id=sample_id, latitude=lat, longitude=lon,
                                           target=target, features_json=features, source="synthetic_csv"))
            session.add_all(rows)
            session.commit()

def load_training_frame() -> pd.DataFrame:
    """Load training records from the database, so the database is the training-data source."""
    with SessionLocal() as session:
        records = session.scalars(select(TrainingSample).order_by(TrainingSample.id)).all()
        rows = []
        for record in records:
            row = dict(record.features_json)
            row["sample_id"] = record.sample_id
            row["latitude"] = record.latitude
            row["longitude"] = record.longitude
            row["flood_occurrence_synthetic"] = record.target
            rows.append(row)
    if not rows:
        raise RuntimeError("No training samples exist in the database.")
    return pd.DataFrame(rows)

def iso(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()

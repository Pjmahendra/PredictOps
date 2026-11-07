"""Database layer (SQLAlchemy 2.0, SQLite by default; PostgreSQL = change DATABASE_URL).

Design: sensor files are NOT copied into the DB (they stay as CSV files on disk and are
referenced by `datasets.stored_path`). The DB stores what the application produces:
runs, per-minute predictions, alert episodes, acknowledgements and model versions.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text,
                        create_engine, event)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Equipment(Base):
    __tablename__ = "equipment"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(primary_key=True)
    equipment_id: Mapped[int] = mapped_column(ForeignKey("equipment.id"))
    original_filename: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(String(500))          # reference, not the data
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    n_rows: Mapped[int] = mapped_column(Integer)
    start_time: Mapped[datetime | None] = mapped_column(DateTime)
    end_time: Mapped[datetime | None] = mapped_column(DateTime)
    validation: Mapped[dict] = mapped_column(JSON)
    source: Mapped[str] = mapped_column(String(20), default="upload")   # upload | sample
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class PredictionRun(Base):
    __tablename__ = "prediction_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id"))
    equipment_id: Mapped[int] = mapped_column(ForeignKey("equipment.id"))
    source: Mapped[str] = mapped_column(String(20))                 # upload | sample | replay | observations
    n_minutes: Mapped[int] = mapped_column(Integer)
    start_time: Mapped[datetime | None] = mapped_column(DateTime)
    end_time: Mapped[datetime | None] = mapped_column(DateTime)
    model_version_detector: Mapped[str] = mapped_column(String(100))
    model_version_anomaly: Mapped[str] = mapped_column(String(100))
    summary: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    predictions = relationship("Prediction", back_populates="run", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("prediction_runs.id", ondelete="CASCADE"))
    equipment_id: Mapped[int] = mapped_column(ForeignKey("equipment.id"))
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    health_score: Mapped[float] = mapped_column(Float)
    health_status: Mapped[str] = mapped_column(String(10))
    failure_risk: Mapped[str] = mapped_column(String(10))
    anomaly_status: Mapped[str] = mapped_column(String(10))
    anomaly_score: Mapped[float] = mapped_column(Float)
    anomaly_severity: Mapped[str] = mapped_column(String(10))
    detector_prob: Mapped[float] = mapped_column(Float)
    reasons: Mapped[list] = mapped_column(JSON)
    recommendations: Mapped[list] = mapped_column(JSON)
    run = relationship("PredictionRun", back_populates="predictions")
    __table_args__ = (Index("ix_pred_equipment_time", "equipment_id", "timestamp"), Index("ix_pred_run", "run_id"))


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("prediction_runs.id", ondelete="CASCADE"), index=True)
    equipment_id: Mapped[int] = mapped_column(ForeignKey("equipment.id"))
    start_time: Mapped[datetime] = mapped_column(DateTime, index=True)
    end_time: Mapped[datetime] = mapped_column(DateTime)
    minutes: Mapped[int] = mapped_column(Integer)
    severity: Mapped[str] = mapped_column(String(10))
    status: Mapped[str] = mapped_column(String(10))
    min_health_score: Mapped[float] = mapped_column(Float)
    reasons: Mapped[list] = mapped_column(JSON)
    recommendations: Mapped[list] = mapped_column(JSON)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime)
    note: Mapped[str] = mapped_column(Text, default="")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50))
    version: Mapped[str] = mapped_column(String(100), unique=True)
    task: Mapped[str] = mapped_column(String(100))
    artifact: Mapped[str] = mapped_column(String(300))
    metrics: Mapped[dict] = mapped_column(JSON)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    registered_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


_engines: dict[str, object] = {}


def get_engine(url: str):
    if url not in _engines:
        eng = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
        if url.startswith("sqlite"):
            @event.listens_for(eng, "connect")
            def _fk(dbapi_conn, _):          # enforce foreign keys + cascades in SQLite
                dbapi_conn.execute("PRAGMA foreign_keys=ON")
        Base.metadata.create_all(eng)
        _engines[url] = eng
    return _engines[url]


def session_factory(url: str):
    return sessionmaker(bind=get_engine(url), expire_on_commit=False)

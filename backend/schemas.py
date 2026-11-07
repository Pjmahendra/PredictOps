"""Pydantic request/response schemas (validated input, structured JSON output)."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class HealthCheck(BaseModel):
    status: str
    models_loaded: bool
    model_versions: dict
    database: str


class ValidationResult(BaseModel):
    ok: bool
    errors: list[str]
    warnings: list[str]
    n_rows: int


class DatasetOut(ORM):
    id: int
    original_filename: str
    sha256: str
    n_rows: int
    start_time: datetime | None
    end_time: datetime | None
    validation: dict
    source: str
    uploaded_at: datetime


class SensorReading(BaseModel):
    """One raw 10-second reading, exactly the columns of the MetroPT-3 CSV."""
    timestamp: datetime
    TP2: float
    TP3: float
    H1: float
    DV_pressure: float
    Reservoirs: float
    Oil_temperature: float
    Motor_current: float
    COMP: int = Field(ge=0, le=1)
    DV_eletric: int = Field(ge=0, le=1)
    Towers: int = Field(ge=0, le=1)
    MPG: int = Field(ge=0, le=1)
    LPS: int = Field(ge=0, le=1)
    Pressure_switch: int = Field(ge=0, le=1)
    Oil_level: int = Field(ge=0, le=1)
    Caudal_impulses: int = Field(ge=0, le=1)

    @field_validator("TP2", "TP3", "H1", "DV_pressure", "Reservoirs")
    @classmethod
    def pressure_range(cls, v):
        if not -1 <= v <= 20:
            raise ValueError("pressure outside physically plausible range [-1, 20] bar")
        return v

    @field_validator("Oil_temperature")
    @classmethod
    def temp_range(cls, v):
        if not -40 <= v <= 150:
            raise ValueError("oil temperature outside plausible range [-40, 150] °C")
        return v

    @field_validator("Motor_current")
    @classmethod
    def current_range(cls, v):
        if not 0 <= v <= 30:
            raise ValueError("motor current outside plausible range [0, 30] A")
        return v


class ObservationsIn(BaseModel):
    """At least ~2 h of history is recommended: features use windows up to 2 h."""
    readings: list[SensorReading] = Field(min_length=6, max_length=20_000)
    store: bool = True


class PredictRequest(BaseModel):
    dataset_id: int | None = None
    sample_name: str | None = None
    start: datetime | None = None
    end: datetime | None = None


class MinuteAssessment(BaseModel):
    timestamp: datetime
    health_score: float
    health_status: str
    failure_risk: str
    anomaly_status: str
    anomaly_score: float
    anomaly_severity: str
    detector_prob: float
    reasons: list[str]
    recommendations: list[str]


class AlertOut(ORM):
    id: int
    run_id: int
    start_time: datetime
    end_time: datetime
    minutes: int
    severity: str
    status: str
    min_health_score: float
    reasons: list
    recommendations: list
    acknowledged: bool
    acknowledged_at: datetime | None
    note: str


class AckIn(BaseModel):
    note: str = Field(default="", max_length=1000)


class RunOut(ORM):
    id: int
    dataset_id: int | None
    source: str
    n_minutes: int
    start_time: datetime | None
    end_time: datetime | None
    model_version_detector: str
    model_version_anomaly: str
    summary: dict
    created_at: datetime


class PredictResponse(BaseModel):
    run: RunOut
    latest: MinuteAssessment
    alerts: list[AlertOut]


class ExplainRequest(BaseModel):
    run_id: int
    timestamp: datetime
    top_k: int = Field(default=6, ge=1, le=20)

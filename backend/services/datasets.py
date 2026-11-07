"""Dataset upload handling: safe storage, size limits, validation, de-duplication."""
from __future__ import annotations

import hashlib
import io
import uuid
from pathlib import Path

import pandas as pd
from fastapi import HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db import Dataset, Equipment
from backend.settings import Settings
from predictops.config import load_config, project_path
from predictops.data.ingestion import validate_frame

ALLOWED_EXT = {".csv"}
DEFAULT_EQUIPMENT = "MetroPT-3 APU compressor"
SAMPLES = {p.name: p for p in project_path("data/sample").glob("*.csv")} if project_path("data/sample").exists() else {}


def default_equipment(db: Session) -> Equipment:
    eq = db.scalar(select(Equipment).where(Equipment.name == DEFAULT_EQUIPMENT))
    if eq is None:
        eq = Equipment(name=DEFAULT_EQUIPMENT, description="Air Production Unit compressor of a metro train (MetroPT-3, UCI)")
        db.add(eq); db.commit()
    return eq


def _read_csv_bytes(data: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(data))
    except pd.errors.EmptyDataError:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "The file is empty.")
    except Exception as ex:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Could not parse CSV: {type(ex).__name__}")


async def save_upload(file: UploadFile, db: Session, settings: Settings) -> tuple[Dataset, pd.DataFrame]:
    name = Path(file.filename or "").name                     # strip any client-supplied directories
    if Path(name).suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only .csv files are accepted.")
    limit = settings.max_upload_mb * 1024 * 1024
    buf = bytearray()
    while chunk := await file.read(1 << 20):
        buf.extend(chunk)
        if len(buf) > limit:
            raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit.")
    data = bytes(buf)
    df = _read_csv_bytes(data)
    rep = validate_frame(df, load_config())
    sha = hashlib.sha256(data).hexdigest()
    up = Path(settings.upload_dir); up.mkdir(parents=True, exist_ok=True)
    stored = up / f"{uuid.uuid4().hex}.csv"                   # server-chosen name: no path traversal
    stored.write_bytes(data)
    ts = pd.to_datetime(df["timestamp"], errors="coerce") if "timestamp" in df else pd.Series(dtype="datetime64[ns]")
    ds = Dataset(equipment_id=default_equipment(db).id, original_filename=name[:255], stored_path=str(stored),
                 sha256=sha, n_rows=len(df), start_time=ts.min() if ts.notna().any() else None,
                 end_time=ts.max() if ts.notna().any() else None, validation=rep.to_dict(), source="upload")
    db.add(ds); db.commit(); db.refresh(ds)
    return ds, df


def save_frame(df: pd.DataFrame, db: Session, settings: Settings, source: str) -> Dataset:
    """Persist an in-memory frame (e.g. JSON observations) as a dataset file + DB row."""
    data = df.to_csv(index=False).encode()
    up = Path(settings.upload_dir); up.mkdir(parents=True, exist_ok=True)
    stored = up / f"{uuid.uuid4().hex}.csv"
    stored.write_bytes(data)
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    ds = Dataset(equipment_id=default_equipment(db).id, original_filename=f"{source}.csv", stored_path=str(stored),
                 sha256=hashlib.sha256(data).hexdigest(), n_rows=len(df), start_time=ts.min(), end_time=ts.max(),
                 validation=validate_frame(df).to_dict(), source=source)
    db.add(ds); db.commit(); db.refresh(ds)
    return ds


def register_sample(name: str, db: Session) -> tuple[Dataset, pd.DataFrame]:
    if name not in SAMPLES:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown sample. Available: {sorted(SAMPLES)}")
    path = SAMPLES[name]
    data = path.read_bytes(); sha = hashlib.sha256(data).hexdigest()
    df = pd.read_csv(path)
    ds = db.scalar(select(Dataset).where(Dataset.sha256 == sha, Dataset.source == "sample"))
    if ds is None:
        ts = pd.to_datetime(df.timestamp)
        ds = Dataset(equipment_id=default_equipment(db).id, original_filename=name, stored_path=str(path), sha256=sha,
                     n_rows=len(df), start_time=ts.min(), end_time=ts.max(),
                     validation=validate_frame(df).to_dict(), source="sample")
        db.add(ds); db.commit(); db.refresh(ds)
    return ds, df


def load_dataset_frame(ds: Dataset) -> pd.DataFrame:
    p = Path(ds.stored_path)
    if not p.exists():
        raise HTTPException(status.HTTP_410_GONE, "The stored file for this dataset no longer exists.")
    return pd.read_csv(p)

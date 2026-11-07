"""FastAPI dependencies: DB session, cached predictor, optional API-key auth."""
from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status

from backend.db import session_factory
from backend.settings import Settings, get_settings
from predictops.inference.predictor import ArtifactError, get_predictor


def get_db(settings: Settings = Depends(get_settings)):
    db = session_factory(settings.database_url)()
    try:
        yield db
    finally:
        db.close()


def predictor_dep():
    try:
        return get_predictor()
    except ArtifactError as ex:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(ex))


def require_api_key(x_api_key: str | None = Header(default=None), settings: Settings = Depends(get_settings)):
    """Authentication for public deployments: active only when API_KEY is set in the environment."""
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Key header")

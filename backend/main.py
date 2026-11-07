"""PredictOps FastAPI application.

Run (from the project root):
    .venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000
Interactive API documentation: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.db import session_factory
from backend.dependencies import require_api_key
from backend.routes.api import router
from backend.services.models_info import register_versions
from backend.settings import get_settings
from predictops.config import project_path

LOG_DIR = project_path("logs"); LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / "api.log")])
log = logging.getLogger("predictops.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    try:
        from predictops.inference.predictor import get_predictor
        p = get_predictor()                                   # load models ONCE at startup
        log.info("models loaded: %s", p.model_versions)
        with session_factory(settings.database_url)() as db:
            register_versions(db)
    except Exception as ex:                                   # app still starts; /health reports 'degraded'
        log.error("startup: models not loaded: %s", ex)
    yield


settings = get_settings()
app = FastAPI(title="PredictOps API", version="1.0.0", lifespan=lifespan,
              description="Equipment health assessment, anomaly detection and failure-signature detection "
                          "for the MetroPT-3 air compressor. Historical data replay — not live industrial monitoring.",
              dependencies=[Depends(require_api_key)])
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_list, allow_methods=["GET", "POST", "DELETE"],
                   allow_headers=["*"])
app.include_router(router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t = time.time()
    resp = await call_next(request)
    log.info("%s %s -> %s (%.0f ms)", request.method, request.url.path, resp.status_code, (time.time() - t) * 1000)
    return resp


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    # never leak internal paths / stack traces to the client
    return JSONResponse(status_code=500, content={"detail": "Internal server error. See server logs."})

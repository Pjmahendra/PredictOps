# PredictOps runtime image (API or dashboard, chosen by the start command).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps -e .
COPY configs ./configs
COPY backend ./backend
COPY frontend ./frontend
COPY models ./models
COPY reports ./reports
COPY data/sample ./data/sample
COPY data/intermediate/metropt3.parquet ./data/intermediate/
COPY data/processed/history_predictions.parquet data/processed/test_predictions.parquet data/processed/features_monitoring.parquet ./data/processed/
RUN useradd -m app && mkdir -p database logs data/uploads && chown -R app /app/database /app/logs /app/data/uploads
USER app
EXPOSE 8000 8501
# default: API. Dashboard: streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

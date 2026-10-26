# OnTime web app: FastAPI backend + frontend in one image.
#
#   docker compose up --build        -> http://localhost:8000
#
# Or without compose:
#   docker build -t ontime .
#   docker run --rm -p 8000:8000 -v ontime-board:/app/var ontime
#
# Optional local AI engine (Ollama) in a second container:
#   docker compose --profile llm up --build
#   docker compose exec ollama ollama pull llama3.2
#
# The image holds the code, the trained model and the prepared app data
# (plus the model-ready table, used only if the model must be rebuilt).
# Raw data and notebooks stay out (see .dockerignore).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ONTIME_MODEL_DIR=/app/models \
    ONTIME_DB=/app/var/ontime.db \
    ONTIME_SETTINGS=/app/var/settings.json \
    ONTIME_HOST=0.0.0.0 \
    ONTIME_PORT=8000

# xgboost wheels need the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY src/ src/
COPY models/ models/
COPY backend/ backend/
COPY frontend/ frontend/
COPY dashboard/*.json dashboard/
COPY data/processed/rolling_backtest_results.csv data/processed/ontime_model_ready.csv data/processed/
COPY tests/ tests/
COPY scripts/ scripts/
COPY score.py run.py ./

RUN useradd --create-home ontime && mkdir -p /app/var && chown -R ontime /app
USER ontime
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"

CMD ["python", "run.py"]

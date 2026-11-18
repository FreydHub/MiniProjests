from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import JSONResponse, Response

from app.bootstrap import build_demo_registry, build_demo_sequence, build_demo_tabular
from app.modeling import ModelStore, load_joblib_model, load_registry, load_sequence_checkpoint

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
LOGS_DIR = ARTIFACTS_DIR / "logs"
DATA_DIR = PROJECT_DIR / "data"
DB_PATH = DATA_DIR / "service.db"
MODEL_REGISTRY_PATH = Path(os.getenv("MODEL_REGISTRY_PATH", MODELS_DIR / "model_registry.json"))
FEATURE_LENGTH = int(os.getenv("FEATURE_LENGTH", "111"))
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", "256"))

MODEL_PATHS = {
    "logreg": Path(os.getenv("LOGREG_MODEL_PATH", MODELS_DIR / "logreg_model.joblib")),
    "random_forest": Path(os.getenv("RF_MODEL_PATH", MODELS_DIR / "random_forest_model.joblib")),
    "xgboost": Path(os.getenv("XGB_MODEL_PATH", MODELS_DIR / "xgboost_model.joblib")),
    "lightgbm": Path(os.getenv("LGBM_MODEL_PATH", MODELS_DIR / "lightgbm_model.joblib")),
    "lstm_improved": Path(os.getenv("LSTM_MODEL_PATH", MODELS_DIR / "lstm_model.pt")),
    "lstm_cnn_lstm": Path(os.getenv("CNN_LSTM_MODEL_PATH", MODELS_DIR / "cnn_lstm_model.pt")),
}

for path in [ARTIFACTS_DIR, MODELS_DIR, LOGS_DIR, DATA_DIR]:
    path.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("kepler_api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(LOGS_DIR / "api.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

REQUEST_COUNT = Counter("kepler_requests_total", "Total requests", ["endpoint", "status", "model_type"])
PREDICTION_COUNT = Counter("kepler_predictions_total", "Total predictions", ["predicted_class", "model_type"])
REQUEST_LATENCY = Histogram("kepler_request_latency_seconds", "Request latency", ["endpoint", "model_type"])
MODEL_LOADED = Gauge("kepler_model_loaded", "Model load state", ["model_type"])
DB_ROWS = Gauge("kepler_logged_requests", "Logged requests in SQLite")

store: ModelStore | None = None


class PredictRequest(BaseModel):
    model_type: str = Field(default="auto")
    feature_vector: list[float] | None = None
    sequence: list[float] | None = None
    kepid: int | None = None
    source: str = "manual"

    @model_validator(mode="after")
    def validate_payload(self):
        if self.feature_vector is None and self.sequence is None:
            raise ValueError("Either feature_vector or sequence must be provided")
        return self


class PredictResponse(BaseModel):
    predicted_label: int
    predicted_class: str
    probability_transit: float
    threshold: float
    model_name: str
    model_type: str
    input_type: str


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS inference_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            kepid INTEGER,
            source TEXT NOT NULL,
            model_type TEXT NOT NULL,
            input_type TEXT NOT NULL,
            predicted_label INTEGER NOT NULL,
            predicted_class TEXT NOT NULL,
            probability_transit REAL NOT NULL,
            threshold REAL NOT NULL,
            latency_ms REAL NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def refresh_db_metric() -> None:
    with get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM inference_logs").fetchone()[0]
        DB_ROWS.set(count)


def initialize_models() -> ModelStore:
    if not MODEL_REGISTRY_PATH.exists():
        build_demo_registry(FEATURE_LENGTH, SEQUENCE_LENGTH, MODEL_REGISTRY_PATH)

    for model_key in ["logreg", "random_forest", "xgboost", "lightgbm"]:
        if not MODEL_PATHS[model_key].exists():
            build_demo_tabular(FEATURE_LENGTH, MODEL_PATHS[model_key])

    if not MODEL_PATHS["lstm_improved"].exists():
        build_demo_sequence(MODEL_PATHS["lstm_improved"], name="lstm_improved")

    registry = load_registry(MODEL_REGISTRY_PATH, FEATURE_LENGTH, SEQUENCE_LENGTH)

    tabular_models = {
        "logreg": load_joblib_model(MODEL_PATHS["logreg"]),
        "random_forest": load_joblib_model(MODEL_PATHS["random_forest"]),
        "xgboost": load_joblib_model(MODEL_PATHS["xgboost"]),
        "lightgbm": load_joblib_model(MODEL_PATHS["lightgbm"]),
    }
    sequence_models = {
        "lstm_improved": load_sequence_checkpoint(MODEL_PATHS["lstm_improved"]),
        "lstm_cnn_lstm": load_sequence_checkpoint(MODEL_PATHS["lstm_cnn_lstm"]),
    }

    for name, model in tabular_models.items():
        MODEL_LOADED.labels(model_type=name).set(1 if model is not None else 0)
    for name, bundle in sequence_models.items():
        MODEL_LOADED.labels(model_type=name).set(1 if bundle is not None else 0)

    logger.info("Loaded models: %s", {k: v is not None for k, v in tabular_models.items()} | {k: v is not None for k, v in sequence_models.items()})
    return ModelStore(tabular_models=tabular_models, sequence_models=sequence_models, registry=registry)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store
    try:
        store = initialize_models()
    except Exception as exc:
        logger.exception("Startup model loading failed: %s", exc)
    refresh_db_metric()
    yield


app = FastAPI(
    title="Kepler Transit Detection Service",
    version="4.0.0",
    description="FastAPI service for selecting tabular and sequence models for Kepler transit detection",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> JSONResponse:
    active = store.available_models() if store is not None else []
    status = "ok" if active else "degraded"
    REQUEST_COUNT.labels(endpoint="/health", status=status, model_type="system").inc()
    return JSONResponse(
        {
            "status": status,
            "available_models": active,
            "default_tabular_model": store.registry.get("default_tabular_model", "xgboost") if store else "xgboost",
            "default_sequence_model": store.registry.get("default_sequence_model", "lstm_improved") if store else "lstm_improved",
            "feature_length": store.feature_length if store else FEATURE_LENGTH,
            "sequence_length": store.sequence_length if store else SEQUENCE_LENGTH,
        }
    )


@app.get("/metrics")
def metrics() -> Response:
    REQUEST_COUNT.labels(endpoint="/metrics", status="ok", model_type="system").inc()
    refresh_db_metric()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/recent")
def recent(limit: int = 10) -> JSONResponse:
    limit = min(max(limit, 1), 100)
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT created_at, kepid, source, model_type, input_type, predicted_label, predicted_class, probability_transit, threshold, latency_ms FROM inference_logs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    REQUEST_COUNT.labels(endpoint="/recent", status="ok", model_type="system").inc()
    return JSONResponse(
        [
            {
                "created_at": row[0],
                "kepid": row[1],
                "source": row[2],
                "model_type": row[3],
                "input_type": row[4],
                "predicted_label": row[5],
                "predicted_class": row[6],
                "probability_transit": row[7],
                "threshold": row[8],
                "latency_ms": row[9],
            }
            for row in rows
        ]
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    start = time.perf_counter()
    if store is None:
        REQUEST_COUNT.labels(endpoint="/predict", status="error", model_type="system").inc()
        raise HTTPException(status_code=503, detail="Models are not loaded")

    chosen = store.choose_model_type(payload.model_type, payload.feature_vector, payload.sequence)

    if chosen in store.sequence_models:
        if payload.sequence is None:
            raise HTTPException(status_code=422, detail=f"sequence is required for {chosen}")
        if len(payload.sequence) != store.sequence_length:
            raise HTTPException(status_code=422, detail=f"Expected {store.sequence_length} sequence values")
    else:
        if payload.feature_vector is None:
            raise HTTPException(status_code=422, detail=f"feature_vector is required for {chosen}")
        if len(payload.feature_vector) != store.feature_length:
            raise HTTPException(status_code=422, detail=f"Expected {store.feature_length} feature values")

    try:
        score, threshold, model_name, input_type = store.predict(chosen, feature_vector=payload.feature_vector, sequence=payload.sequence)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        REQUEST_COUNT.labels(endpoint="/predict", status="error", model_type=chosen).inc()
        raise HTTPException(status_code=500, detail=str(exc))

    label = int(score >= threshold)
    predicted_class = "TRANSIT" if label == 1 else "NO_TRANSIT"
    latency_ms = (time.perf_counter() - start) * 1000.0

    with get_connection() as conn:
        conn.execute(
            "INSERT INTO inference_logs (created_at, kepid, source, model_type, input_type, predicted_label, predicted_class, probability_transit, threshold, latency_ms) VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (payload.kepid, payload.source, chosen, input_type, label, predicted_class, score, threshold, latency_ms),
        )
        conn.commit()
    refresh_db_metric()

    logger.info(
        json.dumps(
            {
                "event": "predict",
                "kepid": payload.kepid,
                "source": payload.source,
                "model_type": chosen,
                "input_type": input_type,
                "predicted_label": label,
                "probability_transit": round(score, 6),
                "threshold": round(threshold, 6),
                "latency_ms": round(latency_ms, 3),
            },
            ensure_ascii=False,
        )
    )
    REQUEST_COUNT.labels(endpoint="/predict", status="ok", model_type=chosen).inc()
    REQUEST_LATENCY.labels(endpoint="/predict", model_type=chosen).observe(latency_ms / 1000.0)
    PREDICTION_COUNT.labels(predicted_class=predicted_class, model_type=chosen).inc()

    return PredictResponse(
        predicted_label=label,
        predicted_class=predicted_class,
        probability_transit=score,
        threshold=threshold,
        model_name=model_name,
        model_type=chosen,
        input_type=input_type,
    )

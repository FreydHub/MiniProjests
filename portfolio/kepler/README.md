# Kepler ML service

FastAPI inference service for classifying transit signals in Kepler time series.

## Components

- FastAPI and Pydantic for request and response validation
- SQLite for inference history
- Streamlit UI
- Prometheus metrics and Docker Compose health checks
- Tabular and sequence model interfaces

## Run the demo

```bash
docker compose up --build
```

Open the UI at http://localhost:8501 and API documentation at http://localhost:8000/docs.
The application can bootstrap demo models when trained artifacts are absent. Demo predictions
are for verifying the service integration. Model training scripts are in `training/`.
Private files, local databases, model weights and raw datasets are excluded from this copy.

## API

- POST /predict
- GET /health
- GET /recent
- GET /metrics

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.modeling import ImprovedSequenceNet


def build_demo_tabular(feature_length: int, path: Path) -> None:
    x = np.vstack([np.zeros((8, feature_length)), np.ones((8, feature_length))])
    y = np.array([0] * 8 + [1] * 8)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500)),
    ])
    model.fit(x, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def build_demo_sequence(path: Path, name: str = "lstm_improved") -> None:
    model = ImprovedSequenceNet(input_size=1, hidden_size=48, num_layers=2, dropout=0.4)
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": {"architecture": "improved", "input_size": 1, "hidden_size": 48, "num_layers": 2, "dropout": 0.4},
        "threshold": 0.5,
        "name": name,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def build_demo_registry(feature_length: int, sequence_length: int, registry_path: Path) -> None:
    registry = {
        "feature_length": feature_length,
        "sequence_length": sequence_length,
        "default_tabular_model": "xgboost",
        "default_sequence_model": "lstm_improved",
        "default_model_type": "xgboost",
        "models": {
            "logreg": {"path": "artifacts/models/logreg_model.joblib", "input": "feature_vector", "threshold": 0.5},
            "random_forest": {"path": "artifacts/models/random_forest_model.joblib", "input": "feature_vector", "threshold": 0.5},
            "xgboost": {"path": "artifacts/models/xgboost_model.joblib", "input": "feature_vector", "threshold": 0.5},
            "lightgbm": {"path": "artifacts/models/lightgbm_model.joblib", "input": "feature_vector", "threshold": 0.5},
            "lstm_improved": {"path": "artifacts/models/lstm_model.pt", "input": "sequence", "threshold": 0.5},
            "lstm_cnn_lstm": {"path": "artifacts/models/cnn_lstm_model.pt", "input": "sequence", "threshold": 0.5},
        },
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)

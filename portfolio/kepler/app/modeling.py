from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn as nn


class ImprovedSequenceNet(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 48, num_layers: int = 2, dropout: float = 0.4):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True,
        )
        self.bn1 = nn.BatchNorm1d(hidden_size * 4)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        mean_pool = out.mean(dim=1)
        max_pool, _ = out.max(dim=1)
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        pooled = self.bn1(pooled)
        return self.head(pooled).squeeze(-1)


class CNNLSTMNet(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 48, dropout: float = 0.4):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=7, padding=3)
        self.bn_conv1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn_conv2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(64, hidden_size, 2, batch_first=True, dropout=dropout, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = torch.relu(self.bn_conv1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn_conv2(self.conv2(x)))
        x = self.pool(x)
        x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.head(h_n).squeeze(-1)


def load_registry(path: Path, feature_length: int = 111, sequence_length: int = 256) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = {}
    registry.setdefault("feature_length", feature_length)
    registry.setdefault("sequence_length", sequence_length)
    registry.setdefault("default_tabular_model", "xgboost")
    registry.setdefault("default_sequence_model", "lstm_improved")
    registry.setdefault("default_model_type", registry["default_tabular_model"])
    registry.setdefault("models", {})
    return registry


def load_joblib_model(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


def load_sequence_checkpoint(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint.get("config", {})
    architecture = config.get("architecture", "improved")
    if architecture == "cnn_lstm":
        model = CNNLSTMNet(
            input_size=config.get("input_size", 1),
            hidden_size=config.get("hidden_size", 48),
            dropout=config.get("dropout", 0.4),
        )
    else:
        model = ImprovedSequenceNet(
            input_size=config.get("input_size", 1),
            hidden_size=config.get("hidden_size", 48),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.4),
        )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return {
        "model": model,
        "threshold": float(checkpoint.get("threshold", 0.5)),
        "name": checkpoint.get("name", path.stem),
        "architecture": architecture,
    }


@dataclass
class ModelStore:
    tabular_models: dict
    sequence_models: dict
    registry: dict

    @property
    def feature_length(self) -> int:
        return int(self.registry.get("feature_length", 111))

    @property
    def sequence_length(self) -> int:
        return int(self.registry.get("sequence_length", 256))

    def available_models(self) -> list[str]:
        out = []
        for name, model in self.tabular_models.items():
            if model is not None:
                out.append(name)
        for name, bundle in self.sequence_models.items():
            if bundle is not None:
                out.append(name)
        return out

    def choose_model_type(self, requested: str, feature_vector, sequence):
        if requested != "auto":
            return requested
        if sequence is not None:
            return self.registry.get("default_sequence_model", "lstm_improved")
        return self.registry.get("default_tabular_model", "xgboost")

    def predict(self, model_type: str, feature_vector=None, sequence=None):
        if model_type in self.sequence_models:
            bundle = self.sequence_models.get(model_type)
            if bundle is None:
                raise RuntimeError(f"Sequence model '{model_type}' is not loaded")
            model = bundle["model"]
            threshold = float(bundle.get("threshold", 0.5))
            x = np.asarray(sequence, dtype=np.float32)[None, :, None]
            with torch.inference_mode():
                logits = model(torch.tensor(x))
                score = float(torch.sigmoid(logits).cpu().numpy()[0])
            return score, threshold, bundle.get("name", model_type), "sequence"

        if model_type in self.tabular_models:
            model = self.tabular_models.get(model_type)
            if model is None:
                raise RuntimeError(f"Tabular model '{model_type}' is not loaded")
            info = self.registry["models"].get(model_type, {})
            threshold = float(info.get("threshold", 0.5))
            x = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
            score = float(model.predict_proba(x)[:, 1][0])
            return score, threshold, model_type, "feature_vector"

        raise RuntimeError(f"Unknown model_type '{model_type}'")

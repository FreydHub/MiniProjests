
from __future__ import annotations
import hashlib
import json
import math
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

import joblib
import lightkurve as lk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm.auto import tqdm
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
LIGHTCURVE_DIR = RAW_DIR / "lightcurves"
FEATURE_CACHE_DIR = DATA_DIR / "feature_cache"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
TABLES_DIR = ARTIFACTS_DIR / "tables"

for path in [DATA_DIR, RAW_DIR, LIGHTCURVE_DIR, FEATURE_CACHE_DIR, ARTIFACTS_DIR, MODELS_DIR, TABLES_DIR]:
    path.mkdir(parents=True, exist_ok=True)

FEATURE_LENGTH = 111
SEQUENCE_LENGTH = 256
N_PER_CLASS = 150
MIN_POINTS = 800
GLOBAL_BINS = 64
LOCAL_BINS = 32
TEST_SIZE = 0.2
VAL_SIZE = 0.25
BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 6
FORCE_REBUILD_DATASET = False
FORCE_REBUILD_FEATURE_CACHE = False
KOI_API_BASE = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
KOI_PARAMS = {
    "table": "cumulative",
    "format": "csv",
    "select": "kepid,kepoi_name,koi_disposition,koi_period,koi_time0bk,koi_duration,koi_depth,koi_prad,koi_impact,koi_teq",
    "where": "koi_disposition in ('CONFIRMED','FALSE POSITIVE')",
}
KOI_URL = KOI_API_BASE + "?" + urlencode(KOI_PARAMS)
KOI_CSV = RAW_DIR / "koi_cumulative_binary.csv"
SAMPLED_CSV = TABLES_DIR / f"sampled_catalog_n{N_PER_CLASS * 2}.csv"
DATASET_CSV = TABLES_DIR / f"hybrid_dataset_n{N_PER_CLASS * 2}.csv"
MODEL_COMPARISON_CSV = TABLES_DIR / "model_comparison.csv"
TRAINING_HISTORY_CSV = TABLES_DIR / "lstm_training_history.csv"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"


class SequenceNet(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
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
        feature_size = hidden_size * 4
        self.head = nn.Sequential(
            nn.Linear(feature_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        mean_pool = out.mean(dim=1)
        max_pool, _ = out.max(dim=1)
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        return self.head(pooled).squeeze(-1)


def ensure_odd(value: int) -> int:
    value = int(value)
    if value % 2 == 0:
        value += 1
    return max(value, 5)


def load_catalog() -> pd.DataFrame:
    if KOI_CSV.exists():
        df = pd.read_csv(KOI_CSV)
    else:
        df = pd.read_csv(KOI_URL)
        df.to_csv(KOI_CSV, index=False)
    df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
    df = df.dropna(subset=["kepid", "koi_period", "koi_time0bk", "koi_duration"])
    df["kepid"] = df["kepid"].astype(int)
    df["label"] = (df["koi_disposition"] == "CONFIRMED").astype(int)
    return df.reset_index(drop=True)


def sample_catalog(df: pd.DataFrame, n_per_class: int = N_PER_CLASS) -> pd.DataFrame:
    if SAMPLED_CSV.exists():
        return pd.read_csv(SAMPLED_CSV)
    parts = []
    for label_name in ["CONFIRMED", "FALSE POSITIVE"]:
        part = df[df["koi_disposition"] == label_name].sample(
            n=min(n_per_class, (df["koi_disposition"] == label_name).sum()),
            random_state=SEED,
        )
        parts.append(part)
    sampled = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    sampled.to_csv(SAMPLED_CSV, index=False)
    return sampled


def record_cache_path(row: pd.Series) -> Path:
    key_src = (
        f"{int(row['kepid'])}|"
        f"{row.get('kepoi_name', '')}|"
        f"{float(row['koi_period']):.8f}|"
        f"{float(row['koi_time0bk']):.8f}|"
        f"{float(row['koi_duration']):.8f}"
    )
    key = hashlib.md5(key_src.encode("utf-8")).hexdigest()
    return FEATURE_CACHE_DIR / f"{key}.json"


def find_local_lightcurve_files(kepid: int) -> list[Path]:
    kid = f"{int(kepid):09d}"
    patterns = [
        f"**/*{kid}*llc.fits",
        f"**/*{kid}*slc.fits",
        f"**/*{kid}*.fits",
    ]
    files = []
    seen = set()
    for pattern in patterns:
        for path in LIGHTCURVE_DIR.glob(pattern):
            resolved = str(path.resolve())
            if path.is_file() and resolved not in seen:
                seen.add(resolved)
                files.append(path)
    return sorted(files)


def load_local_lightcurve(kepid: int):
    files = find_local_lightcurve_files(kepid)
    if not files:
        return None

    curves = []
    for path in files:
        try:
            obj = lk.read(str(path))

            if hasattr(obj, "PDCSAP_FLUX"):
                lc = obj.PDCSAP_FLUX
            elif hasattr(obj, "SAP_FLUX"):
                lc = obj.SAP_FLUX
            else:
                lc = obj

            lc = lc.remove_nans().normalize()
            if len(lc) > 0:
                curves.append(lc)
        except Exception:
            continue

    if not curves:
        return None

    collection = lk.LightCurveCollection(curves)
    stitched = collection.stitch(corrector_func=lambda x: x.remove_nans().normalize())
    stitched = stitched.remove_outliers(sigma=5)

    if len(stitched) < MIN_POINTS:
        return None

    window_length = ensure_odd(min(1001, max(101, len(stitched) // 10)))
    if window_length >= len(stitched):
        window_length = ensure_odd(len(stitched) - 1)

    flattened = stitched.flatten(window_length=window_length, polyorder=2).remove_nans()
    if len(flattened) < MIN_POINTS:
        return None

    return flattened


def download_lightcurve(kepid: int):
    local = load_local_lightcurve(kepid)
    if local is not None:
        return local

    target = f"KIC {int(kepid)}"
    search = lk.search_lightcurve(target, mission="Kepler", author="Kepler")
    if len(search) == 0:
        return None

    collection = search.download_all(download_dir=str(LIGHTCURVE_DIR))
    if collection is None or len(collection) == 0:
        return None

    stitched = collection.stitch(corrector_func=lambda x: x.remove_nans().normalize())
    stitched = stitched.remove_outliers(sigma=5)

    if len(stitched) < MIN_POINTS:
        return None

    window_length = ensure_odd(min(1001, max(101, len(stitched) // 10)))
    if window_length >= len(stitched):
        window_length = ensure_odd(len(stitched) - 1)

    flattened = stitched.flatten(window_length=window_length, polyorder=2).remove_nans()
    if len(flattened) < MIN_POINTS:
        return None

    return flattened


def predownload_lightcurves(df: pd.DataFrame) -> None:
    unique_ids = sorted(df["kepid"].astype(int).unique().tolist())
    for kepid in tqdm(unique_ids, desc="Predownloading lightcurves"):
        if find_local_lightcurve_files(kepid):
            continue
        try:
            _ = download_lightcurve(kepid)
        except Exception:
            pass


def get_cached_record(row: pd.Series) -> dict | None:
    cache_file = record_cache_path(row)

    if cache_file.exists() and not FORCE_REBUILD_FEATURE_CACHE:
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    kepid = int(row["kepid"])

    lc = load_local_lightcurve(kepid)
    if lc is None:
        lc = download_lightcurve(kepid)

    if lc is None:
        return None

    feature_vector, sequence_vector = extract_representations(row, lc)

    cached = {
        "feature_vector": feature_vector.tolist(),
        "sequence_vector": sequence_vector.tolist(),
    }

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cached, f)

    return cached

def bin_series(x: np.ndarray, y: np.ndarray, left: float, right: float, n_bins: int) -> np.ndarray:
    edges = np.linspace(left, right, n_bins + 1)
    ids = np.digitize(x, edges) - 1
    out = np.full(n_bins, np.nan, dtype=float)
    for idx in range(n_bins):
        mask = ids == idx
        if np.any(mask):
            out[idx] = np.nanmean(y[mask])
    fallback = np.nanmedian(y) if np.isfinite(np.nanmedian(y)) else 0.0
    return np.nan_to_num(out, nan=fallback)


def robust_norm(arr: np.ndarray) -> np.ndarray:
    med = np.nanmedian(arr)
    q1 = np.nanpercentile(arr, 25)
    q3 = np.nanpercentile(arr, 75)
    iqr = max(q3 - q1, 1e-6)
    return np.clip((arr - med) / iqr, -5, 5)


def extract_representations(row: pd.Series, lc) -> tuple[np.ndarray, np.ndarray]:
    time_values = np.asarray(lc.time.value if hasattr(lc.time, "value") else lc.time, dtype=float)
    flux_values = np.asarray(lc.flux.value if hasattr(lc.flux, "value") else lc.flux, dtype=float)
    period = float(row["koi_period"])
    epoch = float(row["koi_time0bk"])
    duration_days = float(row["koi_duration"]) / 24.0
    phase_days = ((time_values - epoch + 0.5 * period) % period) - 0.5 * period
    order = np.argsort(phase_days)
    phase_days = phase_days[order]
    flux_values = flux_values[order]

    global_phase = phase_days / period
    transit_window = max(3.0 * duration_days, 0.02 * period)

    global_bins = bin_series(global_phase, flux_values, -0.5, 0.5, GLOBAL_BINS)
    local_bins = bin_series(phase_days, flux_values, -transit_window, transit_window, LOCAL_BINS)

    global_seq = bin_series(global_phase, flux_values, -0.5, 0.5, SEQUENCE_LENGTH // 2)
    local_seq = bin_series(phase_days, flux_values, -transit_window, transit_window, SEQUENCE_LENGTH // 2)
    sequence_vector = np.concatenate([robust_norm(global_seq), robust_norm(local_seq)]).astype(np.float32)

    centered_flux = flux_values - np.nanmedian(flux_values)
    stats = np.array([
        np.nanmean(flux_values),
        np.nanstd(flux_values),
        np.nanmin(flux_values),
        np.nanmax(flux_values),
        np.nanpercentile(flux_values, 5),
        np.nanpercentile(flux_values, 95),
        skew(centered_flux, nan_policy="omit"),
        kurtosis(centered_flux, nan_policy="omit"),
        np.nanmedian(global_bins) - np.nanmin(local_bins),
        (1.0 - np.nanmin(local_bins)) / (np.nanstd(flux_values) + 1e-8),
        duration_days / period,
        float(row["koi_depth"]) if not pd.isna(row["koi_depth"]) else np.nan,
        float(row["koi_prad"]) if not pd.isna(row["koi_prad"]) else np.nan,
        float(row["koi_impact"]) if not pd.isna(row["koi_impact"]) else np.nan,
        float(row["koi_teq"]) if not pd.isna(row["koi_teq"]) else np.nan,
    ], dtype=float)
    feature_vector = np.concatenate([stats, global_bins, local_bins]).astype(np.float32)
    return feature_vector, sequence_vector


def build_hybrid_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if DATASET_CSV.exists() and not FORCE_REBUILD_DATASET:
        return pd.read_csv(DATASET_CSV)

    predownload_lightcurves(df)
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building hybrid dataset"):
        cached = get_cached_record(row)
        if cached is None:
            continue
        records.append(
            {
                "kepid": int(row["kepid"]),
                "kepoi_name": row["kepoi_name"],
                "koi_disposition": row["koi_disposition"],
                "label": int(row["label"]),
                "feature_vector": json.dumps(cached["feature_vector"]),
                "sequence_vector": json.dumps(cached["sequence_vector"]),
            }
        )
    out = pd.DataFrame(records)
    out.to_csv(DATASET_CSV, index=False)
    return out


def decode_dataset(df: pd.DataFrame):
    X_features = np.vstack(df["feature_vector"].apply(json.loads).values).astype(np.float32)
    X_sequences = np.vstack(df["sequence_vector"].apply(json.loads).values).astype(np.float32)
    y = df["label"].astype(int).values
    groups = df["kepid"].astype(int).values
    return X_features, X_sequences, y, groups


def split_groups(y: np.ndarray, groups: np.ndarray):
    dummy = np.zeros((len(y), 1))
    outer = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_val_idx, test_idx = next(outer.split(dummy, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
    train_local, val_local = next(inner.split(dummy[train_val_idx], y[train_val_idx], groups[train_val_idx]))
    train_idx = train_val_idx[train_local]
    val_idx = train_val_idx[val_local]
    return train_idx, val_idx, test_idx


def safe_metric(fn, y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(fn(y_true, y_score))
    except Exception:
        return float("nan")


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.linspace(0.1, 0.9, 81)
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        preds = (scores >= t).astype(int)
        val = f1_score(y_true, preds, zero_division=0)
        if val > best_f1:
            best_f1 = val
            best_t = float(t)
    return best_t


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (scores >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "roc_auc": safe_metric(roc_auc_score, y_true, scores),
        "pr_auc": safe_metric(average_precision_score, y_true, scores),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
    }


def get_tabular_models():
    return {
        "logreg": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", random_state=SEED)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=SEED,
                n_jobs=-1,
            )),
        ]),
        "xgboost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=SEED,
                n_jobs=-1,
                tree_method="hist",
            )),
        ]),
    }


def train_tabular_models(X_train, y_train, X_val, y_val, X_test, y_test):
    rows = []
    fitted = {}
    thresholds = {}
    for name, model in get_tabular_models().items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start
        val_scores = model.predict_proba(X_val)[:, 1]
        test_scores = model.predict_proba(X_test)[:, 1]
        threshold = best_threshold(y_val, val_scores)
        thresholds[name] = threshold
        val_metrics = evaluate_scores(y_val, val_scores, threshold)
        test_metrics = evaluate_scores(y_test, test_scores, threshold)
        rows.append({"model": name, "split": "val", "input_type": "feature_vector", "train_seconds": train_seconds, **val_metrics})
        rows.append({"model": name, "split": "test", "input_type": "feature_vector", "train_seconds": train_seconds, **test_metrics})
        fitted[name] = model
    return fitted, thresholds, rows


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, sampler=None, shuffle=False):
    tensor_x = torch.tensor(X[:, :, None], dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(tensor_x, tensor_y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle if sampler is None else False, sampler=sampler)


def predict_lstm_scores(model, X: np.ndarray, device: str):
    loader = make_loader(X, np.zeros(len(X), dtype=np.float32), batch_size=BATCH_SIZE, shuffle=False)
    scores = []
    model.eval()
    with torch.inference_mode():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            scores.append(probs)
    return np.concatenate(scores)


def train_lstm(X_train, y_train, X_val, y_val, X_test, y_test):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SequenceNet(input_size=1, hidden_size=64, num_layers=2, dropout=0.2).to(device)

    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = np.array([class_weights[int(y)] for y in y_train], dtype=np.float32)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    pos_weight = torch.tensor([max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1.0)], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    train_loader = make_loader(X_train, y_train, batch_size=BATCH_SIZE, sampler=sampler)
    best_state = None
    best_score = -1.0
    patience_left = PATIENCE
    history = []
    start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.item()) * len(xb)

        val_scores = predict_lstm_scores(model, X_val, device)
        threshold = best_threshold(y_val, val_scores)
        val_metrics = evaluate_scores(y_val, val_scores, threshold)
        scheduler.step(val_metrics["pr_auc"])
        epoch_loss = running_loss / max(len(X_train), 1)

        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_threshold": threshold,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        if val_metrics["pr_auc"] > best_score:
            best_score = val_metrics["pr_auc"]
            best_state = {
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "threshold": threshold,
            }
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    train_seconds = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state["state_dict"])
        best_threshold_value = float(best_state["threshold"])
    else:
        best_threshold_value = 0.5

    pd.DataFrame(history).to_csv(TRAINING_HISTORY_CSV, index=False)

    val_scores = predict_lstm_scores(model, X_val, device)
    test_scores = predict_lstm_scores(model, X_test, device)
    rows = [
        {"model": "lstm", "split": "val", "input_type": "sequence", "train_seconds": train_seconds, **evaluate_scores(y_val, val_scores, best_threshold_value)},
        {"model": "lstm", "split": "test", "input_type": "sequence", "train_seconds": train_seconds, **evaluate_scores(y_test, test_scores, best_threshold_value)},
    ]
    checkpoint = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": {"input_size": 1, "hidden_size": 64, "num_layers": 2, "dropout": 0.2},
        "threshold": best_threshold_value,
        "name": "lstm",
    }
    torch.save(checkpoint, MODELS_DIR / "lstm_model.pt")
    return rows, best_threshold_value


def save_registry(tabular_thresholds: dict, lstm_threshold: float, default_tabular_model: str):
    registry = {
        "feature_length": FEATURE_LENGTH,
        "sequence_length": SEQUENCE_LENGTH,
        "default_tabular_model": default_tabular_model,
        "default_sequence_model": "lstm",
        "default_model_type": default_tabular_model,
        "models": {
            "logreg": {"path": str(MODELS_DIR / "logreg_model.joblib"), "input": "feature_vector", "threshold": float(tabular_thresholds["logreg"])},
            "random_forest": {"path": str(MODELS_DIR / "random_forest_model.joblib"), "input": "feature_vector", "threshold": float(tabular_thresholds["random_forest"])},
            "xgboost": {"path": str(MODELS_DIR / "xgboost_model.joblib"), "input": "feature_vector", "threshold": float(tabular_thresholds["xgboost"])},
            "lstm": {"path": str(MODELS_DIR / "lstm_model.pt"), "input": "sequence", "threshold": float(lstm_threshold)},
        },
    }
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def main():
    catalog = load_catalog()
    sampled = sample_catalog(catalog, n_per_class=N_PER_CLASS)
    dataset = build_hybrid_dataset(sampled)
    X_features, X_sequences, y, groups = decode_dataset(dataset)
    train_idx, val_idx, test_idx = split_groups(y, groups)

    Xf_train, Xf_val, Xf_test = X_features[train_idx], X_features[val_idx], X_features[test_idx]
    Xs_train, Xs_val, Xs_test = X_sequences[train_idx], X_sequences[val_idx], X_sequences[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    tabular_models, tabular_thresholds, tabular_rows = train_tabular_models(Xf_train, y_train, Xf_val, y_val, Xf_test, y_test)
    for name, model in tabular_models.items():
        joblib.dump(model, MODELS_DIR / f"{name}_model.joblib")

    lstm_rows, lstm_threshold = train_lstm(Xs_train, y_train, Xs_val, y_val, Xs_test, y_test)

    comparison = pd.DataFrame(tabular_rows + lstm_rows)
    comparison = comparison[["model", "split", "input_type", "threshold", "roc_auc", "pr_auc", "f1", "precision", "recall", "balanced_accuracy", "train_seconds"]]
    comparison.to_csv(MODEL_COMPARISON_CSV, index=False)

    validation_rows = comparison[comparison["split"] == "val"].copy()
    best_tabular = validation_rows[validation_rows["input_type"] == "feature_vector"].sort_values(["pr_auc", "roc_auc", "f1"], ascending=False).iloc[0]["model"]
    save_registry(tabular_thresholds=tabular_thresholds, lstm_threshold=lstm_threshold, default_tabular_model=str(best_tabular))

    summary = {
        "catalog_rows": int(len(catalog)),
        "sampled_rows": int(len(sampled)),
        "dataset_rows": int(len(dataset)),
        "feature_length": FEATURE_LENGTH,
        "sequence_length": SEQUENCE_LENGTH,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "best_tabular_model": str(best_tabular),
        "default_service_model": str(best_tabular),
    }
    with open(ARTIFACTS_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Model comparison table saved to:", MODEL_COMPARISON_CSV)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

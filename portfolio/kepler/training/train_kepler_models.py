from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from urllib.parse import urlencode

import joblib
import lightkurve as lk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm.auto import tqdm
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ==================== КОНСТАНТЫ ====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
LIGHTCURVE_DIR = RAW_DIR / "lightcurves"
FEATURE_CACHE_DIR = DATA_DIR / "feature_cache"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
TABLES_DIR = ARTIFACTS_DIR / "tables"
LOGS_DIR = ARTIFACTS_DIR / "logs"

KOI_API_BASE = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
KOI_PARAMS = {
    "table": "cumulative",
    "format": "csv",
    "select": "kepid,kepoi_name,koi_disposition,koi_period,koi_time0bk,koi_duration,koi_depth,koi_prad,koi_impact,koi_teq",
    "where": "koi_disposition in ('CONFIRMED','FALSE POSITIVE')",
}
KOI_URL = KOI_API_BASE + "?" + urlencode(KOI_PARAMS)

FEATURE_LENGTH = 111
SEQUENCE_LENGTH = 256
GLOBAL_BINS = 64
LOCAL_BINS = 32
MIN_POINTS = 800
TEST_SIZE = 0.2
VAL_SIZE = 0.25
BATCH_SIZE = 32  # Уменьшен для лучшей генерализации
EPOCHS = 50  # Увеличено
PATIENCE = 10  # Увеличено терпение
MAX_WORKERS = 4
ALLOW_NETWORK_DOWNLOAD = True

_LIGHTCURVE_FILE_INDEX: Optional[Dict[str, List[Path]]] = None


# ==================== УТИЛИТЫ ====================
def ensure_dirs() -> None:
    """Создает необходимые директории."""
    for path in [DATA_DIR, RAW_DIR, LIGHTCURVE_DIR, FEATURE_CACHE_DIR,
                 ARTIFACTS_DIR, MODELS_DIR, TABLES_DIR, LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def reset_all_caches() -> None:
    """Очищает все кеши и артефакты."""
    for target in [LIGHTCURVE_DIR, FEATURE_CACHE_DIR, MODELS_DIR, TABLES_DIR, LOGS_DIR]:
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)

    koi_csv = RAW_DIR / "koi_cumulative_binary.csv"
    if koi_csv.exists():
        koi_csv.unlink()

    ensure_dirs()
    print("All caches and artifacts were removed. Fresh run will start from zero.")


def set_paths(n_per_class: int) -> Dict[str, Path]:
    """Возвращает пути к файлам."""
    return {
        "KOI_CSV": RAW_DIR / "koi_cumulative_binary.csv",
        "SAMPLED_CSV": TABLES_DIR / f"sampled_catalog_n{n_per_class * 2}.csv",
        "DATASET_CSV": TABLES_DIR / f"hybrid_dataset_n{n_per_class * 2}.csv",
        "MODEL_COMPARISON_CSV": TABLES_DIR / "model_comparison.csv",
        "TRAINING_HISTORY_CSV": TABLES_DIR / "lstm_training_history.csv",
        "REGISTRY_PATH": MODELS_DIR / "model_registry.json",
    }


@lru_cache(maxsize=1)
def ensure_odd(value: int) -> int:
    """Гарантирует нечетное значение."""
    value = int(value)
    if value % 2 == 0:
        value += 1
    return max(value, 5)


# ==================== УЛУЧШЕННАЯ МОДЕЛЬ НЕЙРОННОЙ СЕТИ ====================
class ImprovedSequenceNet(nn.Module):
    """Улучшенная архитектура с регуляризацией и skip connections."""

    def __init__(self, input_size: int = 1, hidden_size: int = 48,
                 num_layers: int = 2, dropout: float = 0.4):
        super().__init__()

        # Уменьшен hidden_size и увеличен dropout для борьбы с переобучением
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True,
        )

        # Batch normalization для стабильности
        self.bn1 = nn.BatchNorm1d(hidden_size * 4)

        # Более простая голова для предотвращения переобучения
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)

        # Max и mean pooling
        mean_pool = out.mean(dim=1)
        max_pool, _ = out.max(dim=1)
        pooled = torch.cat([mean_pool, max_pool], dim=1)

        # Batch normalization
        pooled = self.bn1(pooled)

        return self.head(pooled).squeeze(-1)


class CNNLSTMNet(nn.Module):
    """Исправленная гибридная CNN-LSTM архитектура."""

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

    def forward(self, x):
        # CNN: [batch, seq_len, 1] → [batch, 1, seq_len]
        x = x.transpose(1, 2)

        x = torch.relu(self.bn_conv1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn_conv2(self.conv2(x)))
        x = self.pool(x)

        # LSTM: [batch, seq_len//4, 64]
        x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)

        # h_n: [4, batch, hidden_size] → берем [2,3] и конкатенируем
        h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)  # [batch, hidden_size*2]

        return self.head(h_n).squeeze(-1)


# ==================== РАБОТА С КАТАЛОГОМ ====================
def load_catalog(koi_csv: Path) -> pd.DataFrame:
    """Загружает каталог KOI."""
    if koi_csv.exists():
        df = pd.read_csv(koi_csv)
    else:
        df = pd.read_csv(KOI_URL)
        df.to_csv(koi_csv, index=False)

    df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
    df = df.dropna(subset=["kepid", "koi_period", "koi_time0bk", "koi_duration"])
    df["kepid"] = df["kepid"].astype(int)
    df["label"] = (df["koi_disposition"] == "CONFIRMED").astype(int)

    return df.reset_index(drop=True)


def sample_catalog(df: pd.DataFrame, sampled_csv: Path, n_per_class: int) -> pd.DataFrame:
    """Создает сбалансированную выборку."""
    if sampled_csv.exists():
        return pd.read_csv(sampled_csv)

    parts = []
    for label_name in ["CONFIRMED", "FALSE POSITIVE"]:
        subset = df[df["koi_disposition"] == label_name]
        n_samples = min(n_per_class, len(subset))
        part = subset.sample(n=n_samples, random_state=SEED)
        parts.append(part)

    sampled = pd.concat(parts, ignore_index=True)
    sampled = sampled.sample(frac=1, random_state=SEED).reset_index(drop=True)
    sampled.to_csv(sampled_csv, index=False)

    return sampled


# ==================== РАБОТА С КРИВЫМИ БЛЕСКА ====================
def build_lightcurve_file_index() -> Dict[str, List[Path]]:
    """Строит индекс файлов кривых блеска для быстрого поиска."""
    index: Dict[str, List[Path]] = {}

    for path in LIGHTCURVE_DIR.rglob("*.fits"):
        keys = set(re.findall(r"(\d{9})", path.name))
        if not keys:
            keys = set(re.findall(r"(\d{9})", path.as_posix()))

        for key in keys:
            index.setdefault(key, []).append(path)

    return index


def get_lightcurve_file_index() -> Dict[str, List[Path]]:
    """Возвращает кешированный индекс файлов."""
    global _LIGHTCURVE_FILE_INDEX

    if _LIGHTCURVE_FILE_INDEX is None:
        _LIGHTCURVE_FILE_INDEX = build_lightcurve_file_index()

    return _LIGHTCURVE_FILE_INDEX


def invalidate_lightcurve_index() -> None:
    """Сбрасывает кеш индекса файлов."""
    global _LIGHTCURVE_FILE_INDEX
    _LIGHTCURVE_FILE_INDEX = None


def find_local_lightcurve_files(kepid: int) -> List[Path]:
    """Находит локальные файлы кривой блеска."""
    index = get_lightcurve_file_index()
    kid = f"{int(kepid):09d}"
    return sorted(index.get(kid, []))


def load_local_lightcurve(kepid: int):
    """Загружает кривую блеска из локальных файлов."""
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

    try:
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

    except Exception:
        return None


def download_lightcurve(kepid: int):
    """Загружает кривую блеска из сети или локально."""
    local = load_local_lightcurve(kepid)
    if local is not None:
        return local

    if not ALLOW_NETWORK_DOWNLOAD:
        return None

    target = f"KIC {int(kepid)}"
    try:
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

        invalidate_lightcurve_index()

        return flattened

    except Exception:
        return None


def predownload_lightcurves(df: pd.DataFrame) -> None:
    """Предварительно загружает кривые блеска."""
    unique_ids = sorted(df["kepid"].astype(int).unique().tolist())
    to_download = [kid for kid in unique_ids if not find_local_lightcurve_files(kid)]

    if not to_download:
        return

    print(f"Pre-downloading {len(to_download)} lightcurves...")

    for kepid in tqdm(to_download, desc="Predownloading lightcurves"):
        try:
            _ = download_lightcurve(kepid)
        except Exception:
            pass


# ==================== ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ====================
def bin_series(x: np.ndarray, y: np.ndarray, left: float, right: float, n_bins: int) -> np.ndarray:
    """Бинирование временного ряда."""
    edges = np.linspace(left, right, n_bins + 1)
    ids = np.digitize(x, edges) - 1
    out = np.full(n_bins, np.nan, dtype=np.float32)

    for idx in range(n_bins):
        mask = ids == idx
        if np.any(mask):
            out[idx] = np.nanmean(y[mask])

    fallback = np.nanmedian(y) if np.isfinite(np.nanmedian(y)) else 0.0
    return np.nan_to_num(out, nan=fallback)


def robust_norm(arr: np.ndarray) -> np.ndarray:
    """Робастная нормализация массива."""
    med = np.nanmedian(arr)
    q1, q3 = np.nanpercentile(arr, [25, 75])
    iqr = max(q3 - q1, 1e-6)
    return np.clip((arr - med) / iqr, -5, 5)


def extract_representations(row: pd.Series, lc) -> Tuple[np.ndarray, np.ndarray]:
    """Извлекает векторы признаков и последовательности."""
    time_values = np.asarray(
        lc.time.value if hasattr(lc.time, "value") else lc.time,
        dtype=np.float32
    )
    flux_values = np.asarray(
        lc.flux.value if hasattr(lc.flux, "value") else lc.flux,
        dtype=np.float32
    )

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

    sequence_vector = np.concatenate([
        robust_norm(global_seq),
        robust_norm(local_seq)
    ]).astype(np.float32)

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
        float(row.get("koi_depth", np.nan)),
        float(row.get("koi_prad", np.nan)),
        float(row.get("koi_impact", np.nan)),
        float(row.get("koi_teq", np.nan)),
    ], dtype=np.float32)

    feature_vector = np.concatenate([stats, global_bins, local_bins]).astype(np.float32)

    return feature_vector, sequence_vector


def record_cache_path(row: pd.Series) -> Path:
    """Возвращает путь к кешированному файлу для записи."""
    key_src = (
        f"{int(row['kepid'])}|"
        f"{row.get('kepoi_name', '')}|"
        f"{float(row['koi_period']):.8f}|"
        f"{float(row['koi_time0bk']):.8f}|"
        f"{float(row['koi_duration']):.8f}"
    )
    key = hashlib.md5(key_src.encode("utf-8")).hexdigest()
    return FEATURE_CACHE_DIR / f"{key}.json"


def get_cached_record(row: pd.Series) -> Optional[Dict]:
    """Получает кешированную запись или создает новую."""
    cache_file = record_cache_path(row)

    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

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

    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cached, f)
    except Exception:
        pass

    return cached


def build_one_record(row_dict: Dict) -> Optional[Dict]:
    """Создает одну запись датасета."""
    row = pd.Series(row_dict)
    cached = get_cached_record(row)

    if cached is None:
        return None

    return {
        "kepid": int(row["kepid"]),
        "kepoi_name": row["kepoi_name"],
        "koi_disposition": row["koi_disposition"],
        "label": int(row["label"]),
        "feature_vector": json.dumps(cached["feature_vector"]),
        "sequence_vector": json.dumps(cached["sequence_vector"]),
    }


def build_hybrid_dataset(df: pd.DataFrame, dataset_csv: Path) -> pd.DataFrame:
    """Строит гибридный датасет с признаками и последовательностями."""
    if dataset_csv.exists():
        return pd.read_csv(dataset_csv)

    predownload_lightcurves(df)

    records = []
    row_dicts = df.to_dict(orient="records")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(build_one_record, row_dict) for row_dict in row_dicts]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Building hybrid dataset"):
            try:
                result = future.result()
                if result is not None:
                    records.append(result)
            except Exception:
                continue

    out = pd.DataFrame(records)
    out.to_csv(dataset_csv, index=False)

    return out


def decode_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Декодирует датасет в numpy массивы."""
    X_features = np.vstack(df["feature_vector"].apply(json.loads).values).astype(np.float32)
    X_sequences = np.vstack(df["sequence_vector"].apply(json.loads).values).astype(np.float32)
    y = df["label"].astype(int).values
    groups = df["kepid"].astype(int).values

    return X_features, X_sequences, y, groups


# ==================== РАЗДЕЛЕНИЕ ДАННЫХ ====================
def split_groups(y: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Разделяет данные по группам."""
    dummy = np.zeros((len(y), 1))

    outer = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_val_idx, test_idx = next(outer.split(dummy, y, groups))

    inner = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
    train_local, val_local = next(inner.split(
        dummy[train_val_idx],
        y[train_val_idx],
        groups[train_val_idx]
    ))

    train_idx = train_val_idx[train_local]
    val_idx = train_val_idx[val_local]

    return train_idx, val_idx, test_idx


# ==================== МЕТРИКИ ====================
def safe_metric(fn, y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Безопасное вычисление метрики."""
    try:
        return float(fn(y_true, y_score))
    except Exception:
        return float("nan")


def best_threshold(y_true: np.ndarray, scores: np.ndarray, metric='f1') -> float:
    """Находит оптимальный порог по заданной метрике."""
    candidates = np.linspace(0.2, 0.8, 61)  # Более разумный диапазон
    best_t, best_val = 0.5, -1.0

    for t in candidates:
        preds = (scores >= t).astype(int)

        if metric == 'f1':
            val = f1_score(y_true, preds, zero_division=0)
        elif metric == 'balanced_accuracy':
            val = balanced_accuracy_score(y_true, preds)
        else:
            val = f1_score(y_true, preds, zero_division=0)

        if val > best_val:
            best_val = val
            best_t = float(t)

    return best_t


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    """Вычисляет набор метрик."""
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


# ==================== ОБУЧЕНИЕ МОДЕЛЕЙ ====================
def train_logreg_rf(X_train, y_train, X_val, y_val, X_test, y_test):
    """Обучает LogReg и Random Forest с улучшенной регуляризацией."""
    models = {
        "logreg": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),  # Более робастный скейлер
            ("clf", LogisticRegression(
                max_iter=5000,
                C=0.1,  # Усилена регуляризация
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
                penalty='l2'
            )),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300,  # Уменьшено для снижения переобучения
                max_depth=10,  # Ограничена глубина
                min_samples_split=10,  # Увеличено
                min_samples_leaf=5,  # Увеличено
                max_features='sqrt',  # Ограничены признаки
                class_weight="balanced_subsample",
                random_state=SEED,
                n_jobs=-1,
            )),
        ]),
    }

    rows = []
    fitted = {}
    thresholds = {}

    for name, model in models.items():
        print(f"Training {name}...")
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start

        val_scores = model.predict_proba(X_val)[:, 1]
        test_scores = model.predict_proba(X_test)[:, 1]

        threshold = best_threshold(y_val, val_scores, metric='balanced_accuracy')
        thresholds[name] = threshold

        rows.append({
            "model": name,
            "split": "val",
            "input_type": "feature_vector",
            "train_seconds": train_seconds,
            **evaluate_scores(y_val, val_scores, threshold)
        })
        rows.append({
            "model": name,
            "split": "test",
            "input_type": "feature_vector",
            "train_seconds": train_seconds,
            **evaluate_scores(y_test, test_scores, threshold)
        })

        fitted[name] = model

    return fitted, thresholds, rows


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int,
                sampler=None, shuffle=False) -> DataLoader:
    """Создает DataLoader для PyTorch."""
    tensor_x = torch.tensor(X[:, :, None], dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(tensor_x, tensor_y)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )


def predict_lstm_scores(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    """Предсказания LSTM модели."""
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


def train_lstm(X_train, y_train, X_val, y_val, X_test, y_test,
               training_history_csv: Path, model_path: Path, architecture='improved'):
    """Обучает LSTM модель с улучшениями."""
    print(f"Training {architecture} LSTM...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Выбор архитектуры
    if architecture == 'cnn_lstm':
        model = CNNLSTMNet(input_size=1, hidden_size=48, dropout=0.4).to(device)
    else:
        model = ImprovedSequenceNet(input_size=1, hidden_size=48, num_layers=2, dropout=0.4).to(device)

    # Взвешенная выборка
    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = np.array([class_weights[int(y)] for y in y_train], dtype=np.float32)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Взвешенная потеря
    pos_weight = torch.tensor([
        max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1.0)
    ], device=device, dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer с меньшим learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    train_loader = make_loader(X_train, y_train, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = make_loader(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

    best_state = None
    best_score = -1.0
    patience_left = PATIENCE
    history = []

    start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        # Обучение
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)

            # L2 регуляризация (дополнительно к weight_decay)
            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss = loss + 1e-5 * l2_reg

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item()) * len(xb)

        # Валидация
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(float(loss.item()) * len(xb))

        val_loss = sum(val_losses) / max(len(X_val), 1)
        val_scores = predict_lstm_scores(model, X_val, device)
        threshold = best_threshold(y_val, val_scores, metric='balanced_accuracy')
        val_metrics = evaluate_scores(y_val, val_scores, threshold)

        scheduler.step()

        epoch_loss = running_loss / max(len(X_train), 1)
        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "val_threshold": threshold,
            **{f"val_{k}": v for k, v in val_metrics.items()}
        })

        print(f"Epoch {epoch}/{EPOCHS} - Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - "
              f"Val PR-AUC: {val_metrics['pr_auc']:.4f} - Val F1: {val_metrics['f1']:.4f}")

        # Early stopping по PR-AUC и Val Loss
        current_score = val_metrics["pr_auc"] - 0.1 * val_loss  # Комбинированная метрика

        if current_score > best_score:
            best_score = current_score
            best_state = {
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "threshold": threshold
            }
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}")
                break

    train_seconds = time.perf_counter() - start

    # Загрузка лучшего состояния
    if best_state is not None:
        model.load_state_dict(best_state["state_dict"])
        best_threshold_value = float(best_state["threshold"])
    else:
        best_threshold_value = 0.5

    # Сохранение истории
    pd.DataFrame(history).to_csv(training_history_csv, index=False)

    # Сохранение модели
    checkpoint = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": {"architecture": architecture, "input_size": 1, "hidden_size": 48, "num_layers": 2, "dropout": 0.4},
        "threshold": best_threshold_value,
        "name": f"lstm_{architecture}",
    }
    torch.save(checkpoint, model_path)

    # Оценка
    val_scores = predict_lstm_scores(model, X_val, device)
    test_scores = predict_lstm_scores(model, X_test, device)

    rows = [
        {
            "model": f"lstm_{architecture}",
            "split": "val",
            "input_type": "sequence",
            "train_seconds": train_seconds,
            **evaluate_scores(y_val, val_scores, best_threshold_value)
        },
        {
            "model": f"lstm_{architecture}",
            "split": "test",
            "input_type": "sequence",
            "train_seconds": train_seconds,
            **evaluate_scores(y_test, test_scores, best_threshold_value)
        },
    ]

    return rows, best_threshold_value


def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """Обучает XGBoost с оптимизированными гиперпараметрами."""
    print("Training XGBoost...")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            min_child_weight=5,
            gamma=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=20,  # Перенесено в конструктор
            random_state=SEED,
            n_jobs=-1,
            tree_method="hist",
        )),
    ])

    start = time.perf_counter()
    model.fit(
        X_train, y_train,
        clf__eval_set=[(X_val, y_val)],
        clf__verbose=False
    )
    train_seconds = time.perf_counter() - start

    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]

    threshold = best_threshold(y_val, val_scores, metric='balanced_accuracy')

    rows = [
        {
            "model": "xgboost",
            "split": "val",
            "input_type": "feature_vector",
            "train_seconds": train_seconds,
            **evaluate_scores(y_val, val_scores, threshold)
        },
        {
            "model": "xgboost",
            "split": "test",
            "input_type": "feature_vector",
            "train_seconds": train_seconds,
            **evaluate_scores(y_test, test_scores, threshold)
        },
    ]

    return model, threshold, rows


def train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test):
    """Обучает LightGBM."""
    print("Training LightGBM...")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            min_child_samples=20,
            num_leaves=15,
            objective="binary",
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )),
    ])

    start = time.perf_counter()
    # LightGBM использует callbacks для early stopping
    model.fit(
        X_train, y_train,
        clf__eval_set=[(X_val, y_val)],
        clf__eval_metric='logloss',
    )
    train_seconds = time.perf_counter() - start

    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]

    threshold = best_threshold(y_val, val_scores, metric='balanced_accuracy')

    rows = [
        {
            "model": "lightgbm",
            "split": "val",
            "input_type": "feature_vector",
            "train_seconds": train_seconds,
            **evaluate_scores(y_val, val_scores, threshold)
        },
        {
            "model": "lightgbm",
            "split": "test",
            "input_type": "feature_vector",
            "train_seconds": train_seconds,
            **evaluate_scores(y_test, test_scores, threshold)
        },
    ]

    return model, threshold, rows


def save_registry(registry_path: Path, tabular_thresholds: Dict,
                  lstm_threshold: float, default_tabular_model: str) -> None:
    """Сохраняет реестр моделей."""
    registry = {
        "feature_length": FEATURE_LENGTH,
        "sequence_length": SEQUENCE_LENGTH,
        "default_tabular_model": default_tabular_model,
        "default_sequence_model": "lstm",
        "default_model_type": default_tabular_model,
        "models": {
            "logreg": {
                "path": str(MODELS_DIR / "logreg_model.joblib"),
                "input": "feature_vector",
                "threshold": float(tabular_thresholds["logreg"])
            },
            "random_forest": {
                "path": str(MODELS_DIR / "random_forest_model.joblib"),
                "input": "feature_vector",
                "threshold": float(tabular_thresholds["random_forest"])
            },
            "xgboost": {
                "path": str(MODELS_DIR / "xgboost_model.joblib"),
                "input": "feature_vector",
                "threshold": float(tabular_thresholds["xgboost"])
            },
            "lightgbm": {
                "path": str(MODELS_DIR / "lightgbm_model.joblib"),
                "input": "feature_vector",
                "threshold": float(tabular_thresholds.get("lightgbm", 0.5))
            },
            "lstm": {
                "path": str(MODELS_DIR / "lstm_model.pt"),
                "input": "sequence",
                "threshold": float(lstm_threshold)
            },
        },
    }

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


# ==================== ОСНОВНЫЕ ФУНКЦИИ ====================
def prepare_dataset_only(paths: Dict[str, Path], n_per_class: int) -> None:
    """Подготовка датасета."""
    print("=" * 80)
    print("STAGE: DATASET PREPARATION")
    print("=" * 80)

    catalog = load_catalog(paths["KOI_CSV"])
    print(f"Loaded catalog: {len(catalog)} objects")

    sampled = sample_catalog(catalog, paths["SAMPLED_CSV"], n_per_class=n_per_class)
    print(f"Sampled catalog: {len(sampled)} objects")

    dataset = build_hybrid_dataset(sampled, paths["DATASET_CSV"])
    print(f"Built dataset: {len(dataset)} objects")

    summary = {
        "catalog_rows": int(len(catalog)),
        "sampled_rows": int(len(sampled)),
        "dataset_rows": int(len(dataset)),
        "feature_length": FEATURE_LENGTH,
        "sequence_length": SEQUENCE_LENGTH,
    }

    with open(ARTIFACTS_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("Dataset prepared successfully")
    print(f"Sampled catalog: {paths['SAMPLED_CSV']}")
    print(f"Hybrid dataset: {paths['DATASET_CSV']}")
    print(f"Rows in dataset: {len(dataset)}")
    print("=" * 80)


def train_models_only(paths: Dict[str, Path]) -> None:
    """Обучение моделей."""
    print("\n" + "=" * 80)
    print("STAGE: MODEL TRAINING")
    print("=" * 80)

    if not paths["DATASET_CSV"].exists():
        raise FileNotFoundError(f"{paths['DATASET_CSV']} not found. Run dataset stage first.")

    dataset = pd.read_csv(paths["DATASET_CSV"])
    print(f"Loaded dataset: {len(dataset)} objects")

    X_features, X_sequences, y, groups = decode_dataset(dataset)
    train_idx, val_idx, test_idx = split_groups(y, groups)

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"Train class distribution: {np.bincount(y[train_idx])}")
    print(f"Val class distribution: {np.bincount(y[val_idx])}")
    print(f"Test class distribution: {np.bincount(y[test_idx])}")

    Xf_train, Xf_val, Xf_test = X_features[train_idx], X_features[val_idx], X_features[test_idx]
    Xs_train, Xs_val, Xs_test = X_sequences[train_idx], X_sequences[val_idx], X_sequences[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    all_rows = []

    # Обучение базовых моделей
    print("\n" + "-" * 80)
    print("Stage 1: Training baseline models (LogReg, Random Forest)")
    print("-" * 80)
    tabular_models, tabular_thresholds, rows_logreg_rf = train_logreg_rf(
        Xf_train, y_train, Xf_val, y_val, Xf_test, y_test
    )
    joblib.dump(tabular_models["logreg"], MODELS_DIR / "logreg_model.joblib")
    joblib.dump(tabular_models["random_forest"], MODELS_DIR / "random_forest_model.joblib")
    all_rows.extend(rows_logreg_rf)

    # Обучение XGBoost
    print("\n" + "-" * 80)
    print("Stage 2: Training XGBoost")
    print("-" * 80)
    xgb_model, xgb_threshold, rows_xgb = train_xgboost(
        Xf_train, y_train, Xf_val, y_val, Xf_test, y_test
    )
    joblib.dump(xgb_model, MODELS_DIR / "xgboost_model.joblib")
    tabular_thresholds["xgboost"] = xgb_threshold
    all_rows.extend(rows_xgb)

    # Обучение LightGBM
    print("\n" + "-" * 80)
    print("Stage 3: Training LightGBM")
    print("-" * 80)
    lgbm_model, lgbm_threshold, rows_lgbm = train_lightgbm(
        Xf_train, y_train, Xf_val, y_val, Xf_test, y_test
    )
    joblib.dump(lgbm_model, MODELS_DIR / "lightgbm_model.joblib")
    tabular_thresholds["lightgbm"] = lgbm_threshold
    all_rows.extend(rows_lgbm)

    # Обучение улучшенного LSTM
    print("\n" + "-" * 80)
    print("Stage 4: Training Improved LSTM")
    print("-" * 80)
    rows_lstm, lstm_threshold = train_lstm(
        Xs_train, y_train, Xs_val, y_val, Xs_test, y_test,
        paths["TRAINING_HISTORY_CSV"], MODELS_DIR / "lstm_model.pt",
        architecture='improved'
    )
    all_rows.extend(rows_lstm)

    # Обучение CNN-LSTM (опционально)
    print("\n" + "-" * 80)
    print("Stage 5: Training CNN-LSTM")
    print("-" * 80)
    rows_cnn_lstm, cnn_lstm_threshold = train_lstm(
        Xs_train, y_train, Xs_val, y_val, Xs_test, y_test,
        TABLES_DIR / "cnn_lstm_training_history.csv",
        MODELS_DIR / "cnn_lstm_model.pt",
        architecture='cnn_lstm'
    )
    all_rows.extend(rows_cnn_lstm)

    # Сохранение результатов
    comparison = pd.DataFrame(all_rows)
    comparison = comparison[[
        "model", "split", "input_type", "threshold",
        "roc_auc", "pr_auc", "f1", "precision", "recall",
        "balanced_accuracy", "train_seconds"
    ]]
    comparison.to_csv(paths["MODEL_COMPARISON_CSV"], index=False)

    # Выбор лучшей модели
    validation_rows = comparison[comparison["split"] == "val"].copy()

    # Лучшая табличная модель по PR-AUC
    best_tabular = validation_rows[validation_rows["input_type"] == "feature_vector"] \
        .sort_values(["pr_auc", "roc_auc", "f1"], ascending=False).iloc[0]["model"]

    # Лучшая sequence модель
    best_sequence = validation_rows[validation_rows["input_type"] == "sequence"] \
        .sort_values(["pr_auc", "roc_auc", "f1"], ascending=False).iloc[0]["model"]

    save_registry(paths["REGISTRY_PATH"], tabular_thresholds, lstm_threshold, str(best_tabular))

    summary = {
        "dataset_rows": int(len(dataset)),
        "feature_length": FEATURE_LENGTH,
        "sequence_length": SEQUENCE_LENGTH,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "best_tabular_model": str(best_tabular),
        "best_sequence_model": str(best_sequence),
        "default_service_model": str(best_tabular),
    }

    with open(ARTIFACTS_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("Model comparison table saved to:", paths["MODEL_COMPARISON_CSV"])
    print("=" * 80)
    print("\n" + comparison.to_string(index=False))
    print("\n" + "=" * 80)
    print(f"\nBest tabular model: {best_tabular}")
    print(f"Best sequence model: {best_sequence}")
    print("=" * 80)


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(description="Exoplanet Classification Pipeline")
    parser.add_argument(
        "--stage",
        choices=["dataset", "train", "all"],
        default="all",
        help="Pipeline stage to run"
    )
    parser.add_argument(
        "--reset-all",
        action="store_true",
        help="Reset all caches and artifacts"
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=150,  # Увеличено для большего датасета
        help="Number of samples per class"
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Disable network downloads"
    )

    args = parser.parse_args()

    global ALLOW_NETWORK_DOWNLOAD
    ALLOW_NETWORK_DOWNLOAD = not args.no_network

    if args.reset_all:
        reset_all_caches()

    ensure_dirs()
    paths = set_paths(args.n_per_class)

    try:
        if args.stage == "dataset":
            prepare_dataset_only(paths, n_per_class=args.n_per_class)
        elif args.stage == "train":
            train_models_only(paths)
        else:  # all
            prepare_dataset_only(paths, n_per_class=args.n_per_class)
            train_models_only(paths)
    except Exception as e:
        print(f"\nERROR: {e}")
        raise


if __name__ == "__main__":
    main()
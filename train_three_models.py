# -*- coding: utf-8 -*-
"""Train RF, XGBoost, and a 3D patch CNN with borehole-wise cross-validation."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

import borehole_split as bsplit
from borehole_split import borehole_kfold_masks, borehole_kfold_table, ensure_labeled_hole_map
from ml_mineral_train import DEFAULT_NEGATIVE_RATIO, DEFAULT_RANDOM_STATE, get_feature_cols

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "results"
LOG: Path | None = None

THR = 0.5
N_SPLITS = 5
ELEVATION_COL = "SurfaceElev"
NOISE_FRAC = 0.05
LOAD_CHUNK = 200000
GRID_DXY = 10.0
GRID_DZ = 10.0
PATCH_RADIUS = 2
CNN_EPOCHS = 80
CNN_BATCH = 64
CNN_INFER_BATCH = 128
CNN_LR = 1.0e-4
CNN_DROPOUT = 0.25
CNN_WEIGHT_DECAY = 1e-3
CNN_CLASS_WEIGHT_STRENGTH = 0.5
CNN_GRAD_CLIP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DSTAR = {"alter_dist": 190.0, "fault_dist": 170.0, "structure_dist": 310.0}
INTENSITY = {
    "fault_dist": (600.0, 170.0),
    "alter_dist": (180.0, 190.0),
    "structure_dist": (180.0, 310.0),
    "contact_dist": (50.0, None),
}
SRC_ORE_POLYGON = 1
SRC_DRILL_POS = 2
SRC_ORE_AND_DRILL = 3
SRC_HARD_NEG = 4
DEM_DROP = {
    "DepthBelowSurface",
    "relative_depth_norm",
    "SurfaceElev",
    "InSurfaceHull",
    "DEM_match_dist_norm",
}
NORM_CLIP = {"relative_depth_norm", "DEM_match_dist_norm"}
NONNEG = {"DepthBelowSurface", "alter_dist", "fault_dist", "structure_dist", "contact_dist"}
RF_PARAMS = dict(
    n_estimators=600,
    max_depth=10,
    min_samples_leaf=2,
    min_samples_split=4,
    max_features="sqrt",
    class_weight="balanced",
    random_state=DEFAULT_RANDOM_STATE,
    n_jobs=1,
)
XGB_PARAMS = dict(
    n_estimators=800,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=5,
    reg_lambda=2.0,
    reg_alpha=0.5,
    scale_pos_weight=1.5,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=DEFAULT_RANDOM_STATE,
    n_jobs=1,
)


def log(msg: str) -> None:
    line = str(msg)
    print(line, flush=True)
    if LOG is not None:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def sample_index(idx: np.ndarray, n: int, seed: int) -> np.ndarray:
    if n <= 0 or len(idx) == 0:
        return np.array([], dtype=np.int64)
    n = min(int(n), len(idx))
    rng = np.random.RandomState(seed)
    if n == len(idx):
        return idx.astype(np.int64, copy=False)
    pick = rng.choice(len(idx), size=n, replace=False)
    return idx[pick].astype(np.int64)


def encode_source(s: pd.Series) -> np.ndarray:
    t = s.astype(str)
    out = np.zeros(len(t), dtype=np.uint8)
    out[t.eq("ore_polygon")] = SRC_ORE_POLYGON
    out[t.eq("drill_au_positive")] = SRC_DRILL_POS
    out[t.eq("ore_polygon+drill_au")] = SRC_ORE_AND_DRILL
    out[t.eq("drill_hard_negative")] = SRC_HARD_NEG
    return out


def far_value_for(dist: np.ndarray) -> float:
    finite = dist[np.isfinite(dist)]
    if len(finite) == 0:
        return 1.0e6
    return float(np.nanpercentile(finite, 99.5) * 1.5)


def build_feature_cache(feature_csv: Path, cache: Path) -> tuple[list[str], int]:
    cache.mkdir(parents=True, exist_ok=True)
    header = pd.read_csv(feature_csv, nrows=0, encoding="utf-8-sig")
    header.columns = [str(c).strip() for c in header.columns]
    feature_cols = get_feature_cols(header)
    n = 0
    for chunk in pd.read_csv(
        feature_csv, usecols=["label"], dtype={"label": "int8"}, chunksize=LOAD_CHUNK, encoding="utf-8-sig"
    ):
        n += len(chunk)
    nfeat = len(feature_cols)
    X = np.memmap(cache / "X_all.dat", dtype=np.float32, mode="w+", shape=(n, nfeat))
    lab = np.memmap(cache / "label.dat", dtype=np.int8, mode="w+", shape=(n,))
    src = np.memmap(cache / "src.dat", dtype=np.uint8, mode="w+", shape=(n,))
    usecols = feature_cols + ["label", "label_source"]
    dtypes = {c: np.float32 for c in feature_cols}
    dtypes["label"] = "int8"
    i0 = 0
    for chunk in pd.read_csv(feature_csv, usecols=usecols, dtype=dtypes, chunksize=LOAD_CHUNK, encoding="utf-8-sig"):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        i1 = i0 + len(chunk)
        X[i0:i1] = chunk[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
        lab[i0:i1] = chunk["label"].to_numpy(np.int8)
        src[i0:i1] = encode_source(chunk["label_source"])
        i0 = i1
    dist_pos = {c: feature_cols.index(c) for c in DSTAR}
    far_used = {}
    for col, d_m in DSTAR.items():
        j = dist_pos[col]
        arr = np.asarray(X[:, j])
        far = far_value_for(arr)
        X[:, j] = np.where(np.isfinite(arr) & (arr <= d_m), arr, far).astype(np.float32, copy=False)
        far_used[col] = far
    X.flush()
    lab.flush()
    src.flush()
    del X, lab, src
    gc.collect()
    (cache / "stamp.json").write_text(
        json.dumps({"dstar": DSTAR, "n": n, "nfeat": nfeat, "far": far_used}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return feature_cols, n


def attach_feature_cache(feature_csv: Path, cache: Path) -> tuple[Path, list[str], int]:
    header = pd.read_csv(feature_csv, nrows=0, encoding="utf-8-sig")
    header.columns = [str(c).strip() for c in header.columns]
    feature_cols = get_feature_cols(header)
    stamp_path = cache / "stamp.json"
    x_path = cache / "X_all.dat"
    if stamp_path.exists() and x_path.exists():
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
        if stamp.get("dstar") == DSTAR and int(stamp.get("nfeat", -1)) == len(feature_cols):
            return cache, feature_cols, int(stamp["n"])
    feature_cols, n = build_feature_cache(feature_csv, cache)
    return cache, feature_cols, n


def open_arrays(cache: Path, n: int, nfeat: int):
    return (
        np.memmap(cache / "X_all.dat", dtype=np.float32, mode="r", shape=(n, nfeat)),
        np.memmap(cache / "label.dat", dtype=np.int8, mode="r", shape=(n,)),
        np.memmap(cache / "src.dat", dtype=np.uint8, mode="r", shape=(n,)),
    )


def lith_dist_indices(cols: list[str]) -> tuple[list[int], list[int], list[int]]:
    lith = [i for i, c in enumerate(cols) if c.startswith("P_")]
    dist = [i for i, c in enumerate(cols) if c.endswith("_dist") and c not in DEM_DROP]
    drop = [i for i, c in enumerate(cols) if c in DEM_DROP]
    return lith, dist, drop


def group_weights(n_lith: int, n_dist: int) -> np.ndarray:
    return np.concatenate(
        [np.full(n_lith, 1.0 / n_lith, dtype=np.float64), np.ones(n_dist, dtype=np.float64)]
    )


def make_xy(lab, src, train_mask, val_mask, seed: int):
    gold = (lab == 1) | np.isin(src, [SRC_ORE_POLYGON, SRC_DRILL_POS, SRC_ORE_AND_DRILL])
    pos_train = np.where(train_mask & (lab == 1))[0]
    hard = np.where(train_mask & (lab == 0) & (src == SRC_HARD_NEG) & (~gold))[0]
    random_pool = np.where(train_mask & (lab == -1) & (~gold))[0]
    if len(pos_train) == 0:
        raise ValueError("No positive training samples.")
    n_neg_target = len(pos_train) * DEFAULT_NEGATIVE_RATIO
    n_hard = min(len(hard), n_neg_target)
    hard_i = sample_index(hard, n_hard, seed)
    rnd_i = sample_index(random_pool, max(0, n_neg_target - n_hard), seed + 1)
    train_i = np.concatenate([pos_train, hard_i, rnd_i])
    y_train = np.concatenate(
        [np.ones(len(pos_train), np.int64), np.zeros(len(hard_i) + len(rnd_i), np.int64)]
    )
    val_i = np.where(val_mask & np.isin(lab, [0, 1]))[0]
    y_val = lab[val_i].astype(np.int64)
    if len(np.unique(y_val)) < 2:
        raise ValueError("Validation fold needs both classes.")
    return train_i, y_train, val_i, y_val


def metrics_at(y, prob, thr: float) -> dict:
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=np.float64)
    if len(np.unique(y)) < 2:
        raise ValueError("Validation fold needs both classes.")
    pred = (prob >= float(thr)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    rec = float(recall_score(y, pred, zero_division=0))
    return {
        "val_n": int(len(y)),
        "val_pos": int((y == 1).sum()),
        "val_neg": int((y == 0).sum()),
        "threshold": float(thr),
        "AUC": float(roc_auc_score(y, prob)),
        "AP": float(average_precision_score(y, prob)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": rec,
        "F1": float(f1_score(y, pred, zero_division=0)),
        "Accuracy": float(accuracy_score(y, pred)),
        "Specificity": spec,
        "Youden": rec + spec - 1.0 if np.isfinite(spec) else float("nan"),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "status": "ok",
        "note": "",
    }


class GroupBalancedRF:
    def __init__(self, weights: np.ndarray, **params):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights = self.weights / self.weights.sum()
        self.n_estimators = int(params.get("n_estimators", 600))
        self.random_state = int(params.get("random_state", 42))
        self.tree_params = dict(
            max_depth=params.get("max_depth", 10),
            min_samples_leaf=params.get("min_samples_leaf", 2),
            min_samples_split=params.get("min_samples_split", 4),
            class_weight=params.get("class_weight", "balanced"),
            max_features=None,
        )
        self.estimators_ = []
        self.feat_idx_ = []

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_feat = X.shape[1]
        k = max(1, int(np.sqrt(n_feat)))
        self.estimators_ = []
        self.feat_idx_ = []
        for i in range(self.n_estimators):
            idx = rng.choice(n_feat, size=k, replace=False, p=self.weights)
            tree = DecisionTreeClassifier(random_state=self.random_state + i, **self.tree_params)
            tree.fit(X[:, idx], y)
            self.estimators_.append(tree)
            self.feat_idx_.append(idx)
        return self

    def predict_proba(self, X):
        acc = None
        for tree, idx in zip(self.estimators_, self.feat_idx_):
            p = tree.predict_proba(X[:, idx])
            acc = p if acc is None else acc + p
        return acc / len(self.estimators_)


def fit_xgb(params: dict, X, y, feature_weights=None):
    kw = dict(params)
    kw["n_jobs"] = -1
    kw["tree_method"] = "hist"
    model = XGBClassifier(**kw)
    if feature_weights is None:
        model.fit(X, y)
    else:
        model.fit(X, y, feature_weights=np.asarray(feature_weights, dtype=np.float32))
    return model


class LithologyMixCNN(nn.Module):
    def __init__(self, n_lith: int, n_dist: int, dropout: float):
        super().__init__()
        self.n_lith = n_lith
        self.lith_mix = nn.Conv3d(n_lith, 1, kernel_size=1)
        in_ch = 1 + n_dist
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        lith = self.lith_mix(x[:, : self.n_lith])
        rest = x[:, self.n_lith :]
        return self.classifier(self.net(torch.cat([lith, rest], dim=1)))


def balanced_class_weights(labels: np.ndarray) -> torch.Tensor:
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    weights = np.ones(2, dtype=np.float32)
    for c in range(2):
        cnt = int((labels == c).sum())
        if cnt > 0:
            weights[c] = n / (2 * cnt)
    s = float(np.clip(CNN_CLASS_WEIGHT_STRENGTH, 0, 1))
    weights = 1.0 + s * (weights - 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_cnn(model, patches, y, seed: int, tag: str):
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=balanced_class_weights(y).to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=CNN_LR, weight_decay=CNN_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CNN_EPOCHS, eta_min=1e-5)
    ds = TensorDataset(torch.from_numpy(patches), torch.from_numpy(y.astype(np.int64)))
    loader = DataLoader(ds, batch_size=CNN_BATCH, shuffle=True, num_workers=0)
    model.train()
    for ep in range(1, CNN_EPOCHS + 1):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CNN_GRAD_CLIP)
            optimizer.step()
        scheduler.step()
        if ep == 1 or ep == CNN_EPOCHS:
            log(f"    [{tag}] epoch {ep:03d}/{CNN_EPOCHS}")
    return model


@torch.no_grad()
def predict_cnn(model, patches) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(patches)),
        batch_size=CNN_INFER_BATCH,
        shuffle=False,
        num_workers=0,
    )
    outs = []
    for (xb,) in loader:
        logits = model(xb.to(DEVICE))
        outs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(outs).astype(np.float64)


def scale_patches(patches, mu, sd):
    return ((patches - mu[:, None, None, None]) / sd[:, None, None, None]).astype(np.float32)


def extract_patch(volume: np.ndarray, ix: int, iy: int, iz: int, radius: int) -> np.ndarray:
    r = int(radius)
    _, nx_, ny_, nz_ = volume.shape
    x0, x1 = ix - r, ix + r + 1
    y0, y1 = iy - r, iy + r + 1
    z0, z1 = iz - r, iz + r + 1
    pad_x = (max(0, -x0), max(0, x1 - nx_))
    pad_y = (max(0, -y0), max(0, y1 - ny_))
    pad_z = (max(0, -z0), max(0, z1 - nz_))
    core = volume[:, max(0, x0) : min(nx_, x1), max(0, y0) : min(ny_, y1), max(0, z0) : min(nz_, z1)]
    if any(pad_x) or any(pad_y) or any(pad_z):
        core = np.pad(core, ((0, 0), pad_x, pad_y, pad_z), mode="edge")
    return core


def gather_patches(volume: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r = PATCH_RADIUS
    c, k = volume.shape[0], 2 * r + 1
    out = np.empty((len(coords), c, k, k, k), dtype=np.float32)
    for i, (x, y, z) in enumerate(coords):
        out[i] = extract_patch(volume, int(x), int(y), int(z), r)
    return out


def grid_coords(rx, ry, rz, n: int):
    gx = np.round(np.asarray(rx, dtype=np.float64) / GRID_DXY) * GRID_DXY
    gy = np.round(np.asarray(ry, dtype=np.float64) / GRID_DXY) * GRID_DXY
    gz = np.round(np.asarray(rz, dtype=np.float64) / GRID_DZ) * GRID_DZ
    _, xi = np.unique(gx, return_inverse=True)
    _, yi = np.unique(gy, return_inverse=True)
    _, zi = np.unique(gz, return_inverse=True)
    coords = np.stack([xi, yi, zi], axis=1).astype(np.int32)
    nx, ny, nz = int(xi.max()) + 1, int(yi.max()) + 1, int(zi.max()) + 1
    return coords, nx, ny, nz


def to_intensity(dist: np.ndarray, decay: float, dstar: float | None) -> np.ndarray:
    dist = np.asarray(dist, dtype=np.float32)
    out = np.exp(-dist / float(decay)).astype(np.float32)
    if dstar is not None:
        out = np.where(dist > float(dstar), 0.0, out).astype(np.float32)
    return out


def apply_intensity(values: np.ndarray, names: list[str]) -> None:
    for j, name in enumerate(names):
        spec = INTENSITY.get(name)
        if spec is None:
            continue
        decay, dstar = spec
        if values.ndim == 2:
            values[:, j] = to_intensity(values[:, j], decay, dstar)
        else:
            values[:, j] = to_intensity(values[:, j], decay, dstar)


def channel_sigma(X_pos: np.ndarray) -> np.ndarray:
    sd = X_pos.std(axis=0).astype(np.float64)
    return NOISE_FRAC * np.where(sd < 1e-6, 0.0, sd)


def jitter_rows(X_pos: np.ndarray, names: list[str], sigma: np.ndarray, rng: np.random.RandomState, n_lith: int) -> np.ndarray:
    out = np.array(X_pos, dtype=np.float32, copy=True)
    n = len(out)
    lith_sig = sigma[:n_lith]
    if np.any(lith_sig > 0):
        lith = out[:, :n_lith] + rng.normal(0.0, lith_sig, size=(n, n_lith)).astype(np.float32)
        lith = np.clip(lith, 0.0, None)
        bad = lith.sum(axis=1) < 1e-6
        lith[bad] = out[bad, :n_lith]
        total = np.maximum(lith.sum(axis=1, keepdims=True), 1e-6)
        out[:, :n_lith] = lith / total
    for j, name in enumerate(names):
        if j < n_lith or name == "InSurfaceHull" or sigma[j] <= 0:
            continue
        out[:, j] = out[:, j] + rng.normal(0.0, sigma[j], size=n).astype(np.float32)
        if name in INTENSITY:
            out[:, j] = np.clip(out[:, j], 0.0, 1.0)
        elif name in NONNEG:
            out[:, j] = np.clip(out[:, j], 0.0, None)
        elif name in NORM_CLIP:
            out[:, j] = np.clip(out[:, j], 0.0, 1.0)
    return out


def jitter_patches(patches: np.ndarray, names: list[str], sigma: np.ndarray, rng: np.random.RandomState, n_lith: int) -> np.ndarray:
    out = np.array(patches, dtype=np.float32, copy=True)
    n = len(out)
    lith_sig = sigma[:n_lith]
    if np.any(lith_sig > 0):
        noise = rng.normal(0.0, 1.0, size=(n, n_lith, 1, 1, 1)).astype(np.float32)
        noise *= lith_sig.reshape(1, n_lith, 1, 1, 1).astype(np.float32)
        lith = np.clip(out[:, :n_lith] + noise, 0.0, None)
        bad = lith.sum(axis=1, keepdims=True)[:, 0, 0, 0, 0] < 1e-6
        lith[bad] = out[bad, :n_lith]
        total = np.maximum(lith.sum(axis=1, keepdims=True), 1e-6)
        out[:, :n_lith] = lith / total
    for j, name in enumerate(names):
        if j < n_lith or name == "InSurfaceHull" or sigma[j] <= 0:
            continue
        noise = rng.normal(0.0, float(sigma[j]), size=(n, 1, 1, 1)).astype(np.float32)
        out[:, j] = out[:, j] + noise
        if name in INTENSITY:
            out[:, j] = np.clip(out[:, j], 0.0, 1.0)
        elif name in NONNEG:
            out[:, j] = np.clip(out[:, j], 0.0, None)
        elif name in NORM_CLIP:
            out[:, j] = np.clip(out[:, j], 0.0, 1.0)
    return out


def main() -> int:
    global LOG
    OUT.mkdir(parents=True, exist_ok=True)
    LOG = OUT / "run.log"
    LOG.write_text("", encoding="utf-8")

    feature_csv = DATA / "feature_table.csv"
    cache = DATA / "feature_cache"
    bsplit.FEATURE_CSV = feature_csv
    bsplit.HOLE_MAP_CSV = DATA / "voxel_hole_id.csv"
    bsplit.DRILL_XLSX = DATA / "parsed_geology.xlsx"
    bsplit.PARSED_EXCEL = DATA / "parsed_geology.xlsx"

    cache, cols, n = attach_feature_cache(feature_csv, cache)
    X0, lab, src = open_arrays(cache, n, len(cols))
    lab, src = np.asarray(lab), np.asarray(src)
    lith_i, dist_i, dem_i = lith_dist_indices(cols)
    dem_keep = [i for i in dem_i if cols[i] == ELEVATION_COL]
    if len(dem_keep) != 1:
        raise RuntimeError(f"Missing column {ELEVATION_COL}.")
    keep = lith_i + dist_i + dem_keep
    names = [cols[i] for i in keep]
    full_names = [cols[i] for i in (lith_i + dist_i + dem_i)]
    vol_ch = [full_names.index(name) for name in names]
    X = np.asarray(X0[:, keep], dtype=np.float32)
    del X0
    apply_intensity(X, names)
    log("distance inputs converted to influence intensity; beyond D* set to 0")
    n_lith, n_dist, n_dem = len(lith_i), len(dist_i), len(dem_keep)
    w = np.concatenate(
        [
            group_weights(n_lith, n_dist),
            np.full(n_dem, 1.0 / max(n_dem, 1), dtype=np.float64),
        ]
    )
    log(f"channels {len(names)}: lithology {n_lith}, distance {n_dist}, elevation 1")

    rx = np.memmap(cache / "RealX.dat", dtype=np.float32, mode="r", shape=(n,))
    ry = np.memmap(cache / "RealY.dat", dtype=np.float32, mode="r", shape=(n,))
    rz = np.memmap(cache / "RealZ.dat", dtype=np.float32, mode="r", shape=(n,))
    hole_map = ensure_labeled_hole_map()
    tab = borehole_kfold_table(hole_map, n_splits=N_SPLITS, seed=DEFAULT_RANDOM_STATE)
    tab.to_csv(OUT / "fold_assignments.csv", index=False, encoding="utf-8-sig")
    coords, nx, ny, nz = grid_coords(rx, ry, rz, n)
    vol_path = DATA / "cnn_volume.dat"
    vol_shape = (len(full_names), nx, ny, nz)
    vol_bytes = int(np.prod(vol_shape, dtype=np.int64) * 4)
    if (not vol_path.exists()) or vol_path.stat().st_size != vol_bytes:
        raise RuntimeError(f"CNN volume size mismatch: {vol_path}")
    vol = np.memmap(vol_path, dtype=np.float32, mode="r", shape=vol_shape)
    models = tuple(sys.argv[1:]) or ("RF", "XGB", "CNN")
    rows = []

    for fold in range(1, N_SPLITS + 1):
        seed = DEFAULT_RANDOM_STATE + 10 * N_SPLITS + fold
        train_mask, val_mask = borehole_kfold_masks(lab, hole_map, fold=fold, fold_table=tab)
        train_i, y_train, val_i, y_val = make_xy(lab, src, train_mask, val_mask, seed)
        if np.intersect1d(train_i, val_i).size:
            raise RuntimeError(f"Fold {fold} has overlapping train and validation voxels.")
        X_tr, X_va = X[train_i], X[val_i]
        pos = y_train == 1
        sigma = channel_sigma(X_tr[pos])
        rng = np.random.RandomState(seed + 17)
        X_aug = np.concatenate([X_tr, jitter_rows(X_tr[pos], names, sigma, rng, n_lith)], axis=0)
        y_aug = np.concatenate([y_train, np.ones(int(pos.sum()), np.int64)])
        log(
            f"fold {fold}: train {len(y_train)} pos {(y_train == 1).sum()}, "
            f"after noise {len(y_aug)}, val {len(y_val)} pos {(y_val == 1).sum()} neg {(y_val == 0).sum()}"
        )

        if "RF" in models:
            rf = GroupBalancedRF(w, **RF_PARAMS)
            rf.random_state = seed
            rf.fit(X_aug, y_aug)
            rec = metrics_at(y_val, rf.predict_proba(X_va)[:, 1], THR)
            rec.update({"fold": fold, "model": "RF"})
            log(f"  RF   AUC={rec['AUC']:.4f}  AP={rec['AP']:.4f}  F1={rec['F1']:.4f}")
            rows.append(rec)
            del rf

        if "XGB" in models:
            xgb = fit_xgb({**XGB_PARAMS, "random_state": seed}, X_aug, y_aug, feature_weights=w)
            rec = metrics_at(y_val, xgb.predict_proba(X_va)[:, 1], THR)
            rec.update({"fold": fold, "model": "XGB"})
            log(f"  XGB  AUC={rec['AUC']:.4f}  AP={rec['AP']:.4f}  F1={rec['F1']:.4f}")
            rows.append(rec)
            del xgb

        if "CNN" in models:
            raw_tr = gather_patches(vol, coords[train_i])[:, vol_ch]
            raw_va = gather_patches(vol, coords[val_i])[:, vol_ch]
            apply_intensity(raw_tr, names)
            apply_intensity(raw_va, names)
            raw_aug = np.concatenate(
                [raw_tr, jitter_patches(raw_tr[pos], names, sigma, rng, n_lith)],
                axis=0,
            )
            mu = X_tr.mean(axis=0)
            sd = np.where(X_tr.std(axis=0) < 1e-6, 1.0, X_tr.std(axis=0))
            cnn = LithologyMixCNN(n_lith, n_dist + n_dem, CNN_DROPOUT)
            cnn = train_one_cnn(cnn, scale_patches(raw_aug, mu, sd), y_aug, seed, f"f{fold}")
            prob = predict_cnn(cnn, scale_patches(raw_va, mu, sd))
            rec = metrics_at(y_val, prob, THR)
            rec.update({"fold": fold, "model": "CNN"})
            log(f"  CNN  AUC={rec['AUC']:.4f}  AP={rec['AP']:.4f}  F1={rec['F1']:.4f}")
            rows.append(rec)
            del cnn, raw_tr, raw_va, raw_aug, prob
            if str(DEVICE).startswith("cuda"):
                torch.cuda.empty_cache()
        del X_tr, X_va, X_aug
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary = (
        df.groupby("model", as_index=False)
        .agg(
            AUC=("AUC", "mean"),
            AP=("AP", "mean"),
            F1=("F1", "mean"),
            Precision=("Precision", "mean"),
            Recall=("Recall", "mean"),
        )
        .sort_values("model")
    )
    summary.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    log("mean over folds:")
    log(summary.to_string(index=False))
    log(f"done: {OUT}")
    del vol
    return 0


if __name__ == "__main__":
    sys.exit(main())

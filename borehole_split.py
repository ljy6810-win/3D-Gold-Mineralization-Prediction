# -*- coding: utf-8 -*-
"""钻孔级训练/验证划分：同一钻孔的样品段不会同时进入训练集和验证集。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

HERE = Path(__file__).resolve().parent


def _resolve_root() -> Path:
    return HERE


ROOT_DIR = _resolve_root()
FEATURE_CSV = ROOT_DIR / "三维地质概率场结果" / "04C_地表DEM约束矿化预测特征表_10m厚度版.csv"
DRILL_XLSX = ROOT_DIR / "钻孔数据库_自动解析.xlsx"
PARSED_EXCEL = ROOT_DIR / "综合三维地质建模结果" / "01_明码文件综合解析结果.xlsx"
HOLE_MAP_CSV = ROOT_DIR / "三维地质概率场结果" / "04C_钻孔级验证_体素HoleID.csv"
HOLE_SPLIT_CSV = ROOT_DIR / "三维地质概率场结果" / "04C_钻孔级验证_孔划分.csv"

SAMPLE_STEP = 5.0
HOLE_MATCH_RADIUS = 12.0
ORE_GROUP = "__ore_polygon__"


def _drill_point(collar_x, collar_y, collar_z, azimuth, dip, depth):
    az = math.radians(float(azimuth))
    dip_rad = math.radians(float(dip))
    horizontal = depth * math.cos(dip_rad)
    dz = depth * math.sin(dip_rad)
    x = collar_x + horizontal * math.sin(az)
    y = collar_y + horizontal * math.cos(az)
    z = collar_z + dz
    return x, y, z


def _sample_interval_depths(start, end, step):
    start = float(start)
    end = float(end)
    if end < start:
        start, end = end, start
    mid = (start + end) / 2.0
    if end - start <= step:
        return np.array([mid], dtype=float)
    arr = np.arange(start, end + 1e-6, step)
    return np.unique(np.concatenate([arr, [mid, end]]))


def read_drill_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in (DRILL_XLSX, PARSED_EXCEL):
        if not path.exists():
            continue
        try:
            collar = pd.read_excel(path, sheet_name="01孔口信息_Collar")
            assay = pd.read_excel(path, sheet_name="04样品化验_Assay")
            collar.columns = [str(c).strip() for c in collar.columns]
            assay.columns = [str(c).strip() for c in assay.columns]
            return collar, assay
        except Exception:
            continue
    return pd.DataFrame(), pd.DataFrame()


def build_assay_points(collar: pd.DataFrame, assay: pd.DataFrame) -> pd.DataFrame:
    need_c = ["HoleID", "X", "Y", "Z", "FinalAzimuth", "FinalDip"]
    need_a = ["HoleID", "From", "To", "Au_value"]
    if collar.empty or assay.empty:
        return pd.DataFrame()
    for c in need_c:
        if c not in collar.columns:
            return pd.DataFrame()
    for c in need_a:
        if c not in assay.columns:
            return pd.DataFrame()

    hole_info = collar.dropna(subset=["HoleID", "X", "Y", "Z"]).copy()
    hole_info["HoleID"] = hole_info["HoleID"].astype(str).str.strip()
    hole_info = hole_info.set_index("HoleID")

    assay = assay.copy()
    assay["HoleID"] = assay["HoleID"].astype(str).str.strip()
    assay["Au_value"] = pd.to_numeric(assay["Au_value"], errors="coerce")
    assay["From"] = pd.to_numeric(assay["From"], errors="coerce")
    assay["To"] = pd.to_numeric(assay["To"], errors="coerce")
    assay = assay.dropna(subset=["HoleID", "From", "To", "Au_value"])

    rows = []
    for r in assay.itertuples():
        hid = str(r.HoleID)
        if hid not in hole_info.index:
            continue
        h = hole_info.loc[hid]
        if isinstance(h, pd.DataFrame):
            h = h.iloc[0]
        az = h.get("FinalAzimuth")
        dip = h.get("FinalDip")
        if pd.isna(az):
            az = 0.0
        if pd.isna(dip):
            dip = -90.0
        for d in _sample_interval_depths(r.From, r.To, SAMPLE_STEP):
            px, py, pz = _drill_point(h["X"], h["Y"], h["Z"], az, dip, d)
            rows.append(
                {
                    "HoleID": hid,
                    "Au_value": float(r.Au_value),
                    "sample_RealX": px,
                    "sample_RealY": py,
                    "sample_RealZ": pz,
                }
            )
    return pd.DataFrame(rows)


def _load_labeled_voxels(feature_csv: Path) -> pd.DataFrame:
    """只读已标注体素，保留原始行号作为 voxel_index。"""
    usecols = ["label", "label_source", "RealX", "RealY", "RealZ"]
    parts = []
    i0 = 0
    for chunk in pd.read_csv(feature_csv, encoding="utf-8-sig", usecols=usecols, chunksize=250000):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        mask = chunk["label"].isin([0, 1]).to_numpy()
        if mask.any():
            hit = chunk.loc[mask].copy()
            hit.index = np.flatnonzero(mask) + i0
            parts.append(hit)
        i0 += len(chunk)
    if not parts:
        raise ValueError("特征表中没有已标注体素。")
    return pd.concat(parts)


def _labeled_frame_from_df(df: pd.DataFrame) -> pd.DataFrame:
    need = ["label", "RealX", "RealY", "RealZ"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"特征表缺少字段：{missing}")
    lab = df.loc[df["label"].isin([0, 1]), need + (["label_source"] if "label_source" in df.columns else [])].copy()
    if "label_source" not in lab.columns:
        lab["label_source"] = ""
    return lab


def _labeled_frame_from_arrays(
    xyz: np.ndarray,
    label: np.ndarray,
    src: np.ndarray | None,
    voxel_index: np.ndarray,
) -> pd.DataFrame:
    lab = pd.DataFrame(
        {
            "RealX": np.asarray(xyz)[:, 0],
            "RealY": np.asarray(xyz)[:, 1],
            "RealZ": np.asarray(xyz)[:, 2],
            "label": np.asarray(label).astype(int),
            "label_source": "" if src is None else np.asarray(src).astype(str),
        },
        index=np.asarray(voxel_index, dtype=np.int64),
    )
    return lab[lab["label"].isin([0, 1])]


def _map_voxels_to_holes(labeled: pd.DataFrame, radius: float = HOLE_MATCH_RADIUS) -> pd.DataFrame:
    collar, assay = read_drill_tables()
    pts = build_assay_points(collar, assay)
    if pts.empty:
        raise ValueError("无法从钻孔数据库构建样品点，无法做钻孔级划分。")

    tree = cKDTree(pts[["sample_RealX", "sample_RealY", "sample_RealZ"]].values.astype(float))
    xyz = labeled[["RealX", "RealY", "RealZ"]].values.astype(float)
    neighbors = tree.query_ball_point(xyz, r=float(radius))
    hole_ids = pts["HoleID"].astype(str).to_numpy()

    rows = []
    for i, (vid, rec) in enumerate(labeled.iterrows()):
        nbs = neighbors[i]
        if not nbs:
            dist, j = tree.query(xyz[i])
            if float(dist) <= 20.0:
                nbs = [int(j)]
        if nbs:
            uniq, cnts = np.unique(hole_ids[nbs], return_counts=True)
            for hid, n_hits in zip(uniq.tolist(), cnts.tolist()):
                rows.append(
                    {
                        "voxel_index": int(vid),
                        "HoleID": str(hid),
                        "n_hits": int(n_hits),
                        "label": int(rec["label"]),
                        "label_source": str(rec.get("label_source", "")),
                        "RealX": float(rec["RealX"]),
                        "RealY": float(rec["RealY"]),
                        "RealZ": float(rec["RealZ"]),
                    }
                )
        else:
            rows.append(
                {
                    "voxel_index": int(vid),
                    "HoleID": "",
                    "n_hits": 0,
                    "label": int(rec["label"]),
                    "label_source": str(rec.get("label_source", "")),
                    "RealX": float(rec["RealX"]),
                    "RealY": float(rec["RealY"]),
                    "RealZ": float(rec["RealZ"]),
                }
            )
    return pd.DataFrame(rows)


def _map_is_fresh(map_csv: Path, feature_csv: Path) -> bool:
    if not map_csv.exists():
        return False
    if not feature_csv.exists():
        return True
    return map_csv.stat().st_mtime >= feature_csv.stat().st_mtime


def ensure_labeled_hole_map(
    df: pd.DataFrame | None = None,
    *,
    xyz: np.ndarray | None = None,
    label: np.ndarray | None = None,
    src: np.ndarray | None = None,
    voxel_index: np.ndarray | None = None,
    feature_csv: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """已标注体素 ↔ HoleID（一体素可对应多孔）。无钻孔命中的矿体多边形体素 HoleID 为空。"""
    feature_csv = Path(feature_csv or FEATURE_CSV)
    if (not force) and _map_is_fresh(HOLE_MAP_CSV, feature_csv):
        out = pd.read_csv(HOLE_MAP_CSV, encoding="utf-8-sig")
        out["HoleID"] = out["HoleID"].fillna("").astype(str)
        return out

    if df is not None:
        labeled = _labeled_frame_from_df(df)
    elif xyz is not None and label is not None and voxel_index is not None:
        labeled = _labeled_frame_from_arrays(xyz, label, src, voxel_index)
    else:
        labeled = _load_labeled_voxels(feature_csv)

    hole_map = _map_voxels_to_holes(labeled)
    HOLE_MAP_CSV.parent.mkdir(parents=True, exist_ok=True)
    hole_map.to_csv(HOLE_MAP_CSV, index=False, encoding="utf-8-sig")
    return hole_map


def _hole_stats(hole_map: pd.DataFrame) -> pd.DataFrame:
    drill = hole_map[hole_map["HoleID"].astype(str).str.len() > 0].copy()
    if drill.empty:
        raise ValueError("已标注体素未能匹配到任何钻孔。")
    rows = []
    for hid, sub in drill.groupby("HoleID"):
        vox = sub.drop_duplicates("voxel_index")
        rows.append(
            {
                "HoleID": str(hid),
                "n_labeled": int(len(vox)),
                "n_pos": int((vox["label"] == 1).sum()),
                "n_neg": int((vox["label"] == 0).sum()),
            }
        )
    stats = pd.DataFrame(rows).sort_values("HoleID").reset_index(drop=True)
    stats["has_pos"] = (stats["n_pos"] > 0).astype(int)
    return stats


def _split_hole_ids(
    stats: pd.DataFrame,
    *,
    val_fraction: float,
    seed: int,
    test_fraction: float = 0.0,
) -> tuple[set[str], set[str], set[str]]:
    holes = stats["HoleID"].astype(str).to_numpy()
    y = stats["has_pos"].to_numpy(dtype=int)
    if len(holes) < 2:
        raise ValueError("钻孔数过少，无法做钻孔级验证。")

    def _split(ids, labels, test_size, rnd):
        ids = np.asarray(ids)
        labels = np.asarray(labels, dtype=int)
        n_val = max(1, int(round(len(ids) * test_size))) if test_size > 0 else 0
        if n_val <= 0:
            return ids, np.array([], dtype=ids.dtype)
        n_val = min(n_val, len(ids) - 1)
        size = n_val / len(ids)
        can_strat = labels.min() != labels.max() and labels.sum() >= 2 and (len(labels) - labels.sum()) >= 2
        try:
            tr, va = train_test_split(
                ids,
                test_size=size,
                random_state=rnd,
                stratify=labels if can_strat else None,
            )
        except ValueError:
            tr, va = train_test_split(ids, test_size=size, random_state=rnd)
        return tr, va

    if test_fraction > 0:
        y_map = dict(zip(holes.tolist(), y.tolist()))
        tv, te = _split(holes, y, test_fraction, seed)
        y_tv = np.array([y_map[str(h)] for h in tv], dtype=int)
        rel_val = val_fraction / max(val_fraction + (1.0 - val_fraction - test_fraction), 1e-9)
        tr, va = _split(tv, y_tv, rel_val, seed + 1)
        return set(map(str, tr)), set(map(str, va)), set(map(str, te))

    tr, va = _split(holes, y, val_fraction, seed)
    return set(map(str, tr)), set(map(str, va)), set()


def _voxel_groups(hole_map: pd.DataFrame) -> dict[int, set[str]]:
    groups: dict[int, set[str]] = {}
    for r in hole_map.itertuples():
        vid = int(r.voxel_index)
        hid = str(r.HoleID).strip()
        groups.setdefault(vid, set())
        if hid:
            groups[vid].add(hid)
    return groups


def _assign_voxels(
    hole_map: pd.DataFrame,
    val_holes: set[str],
    test_holes: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    test_holes = test_holes or set()
    groups = _voxel_groups(hole_map)
    val_idx, test_idx, train_idx = [], [], []
    for vid, hs in groups.items():
        if hs & test_holes:
            test_idx.append(vid)
        elif hs & val_holes:
            val_idx.append(vid)
        else:
            train_idx.append(vid)
    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


def _move_hole_if_missing_class(
    hole_map: pd.DataFrame,
    train_holes: set[str],
    val_holes: set[str],
    *,
    need_pos: bool,
) -> tuple[set[str], set[str]]:
    """若验证孔缺少正类或负类，从训练孔挪一口补上。"""
    stats = _hole_stats(hole_map)
    col = "n_pos" if need_pos else "n_neg"
    cand = stats[stats["HoleID"].isin(train_holes) & (stats[col] > 0)]
    if cand.empty:
        return train_holes, val_holes
    hid = str(cand.sort_values(col, ascending=False).iloc[0]["HoleID"])
    train_holes = set(train_holes)
    val_holes = set(val_holes)
    train_holes.discard(hid)
    val_holes.add(hid)
    return train_holes, val_holes


def _write_split_table(
    stats: pd.DataFrame,
    train_holes: set[str],
    val_holes: set[str],
    test_holes: set[str],
    hole_map: pd.DataFrame,
) -> pd.DataFrame:
    tab = stats.copy()
    tab["split"] = tab["HoleID"].map(
        lambda h: "test" if h in test_holes else ("val" if h in val_holes else "train")
    )
    ore = hole_map[hole_map["HoleID"].astype(str).str.len() == 0]
    if not ore.empty:
        vox = ore.drop_duplicates("voxel_index")
        tab = pd.concat(
            [
                tab,
                pd.DataFrame(
                    [
                        {
                            "HoleID": ORE_GROUP,
                            "n_labeled": int(len(vox)),
                            "n_pos": int((vox["label"] == 1).sum()),
                            "n_neg": int((vox["label"] == 0).sum()),
                            "has_pos": int((vox["label"] == 1).any()),
                            "split": "train",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    tab = tab.sort_values(["split", "HoleID"]).reset_index(drop=True)
    HOLE_SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(HOLE_SPLIT_CSV, index=False, encoding="utf-8-sig")
    return tab


def summarize_hole_split(split_tab: pd.DataFrame, val_n: int, val_pos: int, val_neg: int) -> str:
    holes = split_tab[split_tab["HoleID"].astype(str) != ORE_GROUP]
    n_train_h = int((holes["split"] == "train").sum())
    n_val_h = int((holes["split"] == "val").sum())
    n_test_h = int((holes["split"] == "test").sum())
    val_ids = ", ".join(holes.loc[holes["split"] == "val", "HoleID"].astype(str))
    bits = [
        f"钻孔级划分：训练孔 {n_train_h}，验证孔 {n_val_h}",
    ]
    if n_test_h:
        bits.append(f"测试孔 {n_test_h}")
    bits.append(f"验证体素 {val_n} (pos={val_pos} neg={val_neg})")
    bits.append(f"验证孔：{val_ids}")
    return "；".join(bits)


def borehole_train_val_masks(
    df: pd.DataFrame,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """按钻孔留出验证：验证孔上的体素全部进验证集；未挂孔的矿体多边形体素留在训练。"""
    hole_map = ensure_labeled_hole_map(df)
    stats = _hole_stats(hole_map)
    train_holes, val_holes, test_holes = _split_hole_ids(stats, val_fraction=val_fraction, seed=seed)
    train_idx, val_idx, _ = _assign_voxels(hole_map, val_holes, test_holes)

    labeled = hole_map.drop_duplicates("voxel_index")
    val_lab = labeled[labeled["voxel_index"].isin(val_idx)]
    if int((val_lab["label"] == 1).sum()) == 0:
        train_holes, val_holes = _move_hole_if_missing_class(hole_map, train_holes, val_holes, need_pos=True)
        train_idx, val_idx, _ = _assign_voxels(hole_map, val_holes, test_holes)
        val_lab = labeled[labeled["voxel_index"].isin(val_idx)]
    if int((val_lab["label"] == 0).sum()) == 0:
        train_holes, val_holes = _move_hole_if_missing_class(hole_map, train_holes, val_holes, need_pos=False)
        train_idx, val_idx, _ = _assign_voxels(hole_map, val_holes, test_holes)
        val_lab = labeled[labeled["voxel_index"].isin(val_idx)]

    if val_lab.empty or val_lab["label"].nunique() < 2:
        raise ValueError("钻孔级划分后验证集缺少正负两类，请检查各孔标签分布。")

    split_tab = _write_split_table(stats, train_holes, val_holes, test_holes, hole_map)
    val_mask = df.index.isin(val_idx)
    train_mask = ~val_mask
    print(
        summarize_hole_split(
            split_tab,
            val_n=int(len(val_lab)),
            val_pos=int((val_lab["label"] == 1).sum()),
            val_neg=int((val_lab["label"] == 0).sum()),
        ),
        flush=True,
    )
    return train_mask, val_mask


def borehole_train_val_test_masks(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """按钻孔三划分（默认 60/20/20 孔）。"""
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train/val/test 比例之和必须为 1。")
    hole_map = ensure_labeled_hole_map(df)
    stats = _hole_stats(hole_map)
    train_holes, val_holes, test_holes = _split_hole_ids(
        stats, val_fraction=val_frac, seed=seed, test_fraction=test_frac
    )
    train_idx, val_idx, test_idx = _assign_voxels(hole_map, val_holes, test_holes)
    labeled = hole_map.drop_duplicates("voxel_index")
    val_lab = labeled[labeled["voxel_index"].isin(val_idx)]
    te_lab = labeled[labeled["voxel_index"].isin(test_idx)]
    if val_lab.empty or val_lab["label"].nunique() < 2:
        raise ValueError("钻孔级三划分后验证集缺少正负两类。")
    if te_lab.empty or te_lab["label"].nunique() < 2:
        raise ValueError("钻孔级三划分后测试集缺少正负两类。")

    split_tab = _write_split_table(stats, train_holes, val_holes, test_holes, hole_map)
    print(
        summarize_hole_split(
            split_tab,
            val_n=int(len(val_lab)),
            val_pos=int((val_lab["label"] == 1).sum()),
            val_neg=int((val_lab["label"] == 0).sum()),
        )
        + f"；测试体素 {len(te_lab)} (pos={(te_lab['label']==1).sum()} neg={(te_lab['label']==0).sum()})",
        flush=True,
    )
    val_mask = pd.Series(df.index.isin(val_idx), index=df.index)
    test_mask = pd.Series(df.index.isin(test_idx), index=df.index)
    train_mask = ~(val_mask | test_mask)
    return train_mask, val_mask, test_mask


def hole_codes_array(n: int, hole_map: pd.DataFrame) -> tuple[np.ndarray, dict[int, str]]:
    """供 memmap 流程使用：每个体素一个主孔编码，-1 表示未挂孔。多孔体素取命中最多的孔。"""
    codes = np.full(int(n), -1, dtype=np.int32)
    drill = hole_map[hole_map["HoleID"].astype(str).str.len() > 0].copy()
    if drill.empty:
        return codes, {}
    drill = drill.sort_values(["voxel_index", "n_hits"], ascending=[True, False])
    primary = drill.drop_duplicates("voxel_index", keep="first")
    holes = sorted(primary["HoleID"].astype(str).unique())
    name_to_code = {h: i for i, h in enumerate(holes)}
    codes[primary["voxel_index"].to_numpy(dtype=np.int64)] = (
        primary["HoleID"].astype(str).map(name_to_code).to_numpy(dtype=np.int32)
    )
    return codes, {i: h for h, i in name_to_code.items()}


def borehole_masks_from_label(
    label: np.ndarray,
    hole_map: pd.DataFrame,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """numpy 版钻孔级划分，返回与 label 等长的 bool mask。"""
    stats = _hole_stats(hole_map)
    train_holes, val_holes, test_holes = _split_hole_ids(stats, val_fraction=val_fraction, seed=seed)
    _, val_idx, _ = _assign_voxels(hole_map, val_holes, test_holes)
    labeled = hole_map.drop_duplicates("voxel_index")
    val_lab = labeled[labeled["voxel_index"].isin(val_idx)]
    if int((val_lab["label"] == 1).sum()) == 0:
        train_holes, val_holes = _move_hole_if_missing_class(hole_map, train_holes, val_holes, need_pos=True)
        _, val_idx, _ = _assign_voxels(hole_map, val_holes, test_holes)
        val_lab = labeled[labeled["voxel_index"].isin(val_idx)]
    if int((val_lab["label"] == 0).sum()) == 0:
        train_holes, val_holes = _move_hole_if_missing_class(hole_map, train_holes, val_holes, need_pos=False)
        _, val_idx, _ = _assign_voxels(hole_map, val_holes, test_holes)
        val_lab = labeled[labeled["voxel_index"].isin(val_idx)]
    if val_lab.empty or val_lab["label"].nunique() < 2:
        raise ValueError("钻孔级划分后验证集缺少正负两类。")
    _write_split_table(stats, train_holes, val_holes, set(), hole_map)
    val_mask = np.zeros(len(label), dtype=bool)
    val_mask[np.asarray(val_idx, dtype=np.int64)] = True
    train_mask = ~val_mask
    return train_mask, val_mask


def borehole_kfold_table(
    hole_map: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """按孔随机五折（分层 has_pos）。同一孔只进一折。不写 04C 划分文件。"""
    stats = _hole_stats(hole_map)
    if len(stats) < n_splits:
        raise ValueError(f"钻孔数 {len(stats)} 少于折数 {n_splits}。")
    y = stats["has_pos"].to_numpy(dtype=int)
    fold_id = np.zeros(len(stats), dtype=int)
    n_pos_h = int(y.sum())
    n_neg_h = int(len(y) - n_pos_h)
    can_strat = y.min() != y.max() and n_pos_h >= n_splits and n_neg_h >= n_splits
    splitter = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        if can_strat
        else KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    )
    for i, (_, va) in enumerate(splitter.split(np.arange(len(stats)), y), start=1):
        fold_id[va] = i
    if int((fold_id < 1).sum()) > 0:
        raise ValueError("钻孔五折未能覆盖全部孔。")
    out = stats.copy()
    out["HoleID"] = out["HoleID"].astype(str)
    out["fold"] = fold_id
    return out.sort_values(["fold", "HoleID"]).reset_index(drop=True)


def borehole_kfold_masks(
    label: np.ndarray,
    hole_map: pd.DataFrame,
    *,
    fold: int,
    fold_table: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """留出指定折的钻孔为验证集。未挂孔矿体多边形留在训练。不改 04C 划分文件。"""
    fold = int(fold)
    val_holes = set(fold_table.loc[fold_table["fold"] == fold, "HoleID"].astype(str))
    if not val_holes:
        raise ValueError(f"折 {fold} 没有钻孔。")
    _, val_idx, _ = _assign_voxels(hole_map, val_holes, set())
    labeled = hole_map.drop_duplicates("voxel_index")
    val_lab = labeled[labeled["voxel_index"].isin(val_idx)]
    n_pos = int((val_lab["label"] == 1).sum())
    n_neg = int((val_lab["label"] == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"折 {fold} 验证集缺少正或负类（pos={n_pos} neg={n_neg}）。")
    val_mask = np.zeros(len(label), dtype=bool)
    val_mask[np.asarray(val_idx, dtype=np.int64)] = True
    train_mask = ~val_mask
    return train_mask, val_mask

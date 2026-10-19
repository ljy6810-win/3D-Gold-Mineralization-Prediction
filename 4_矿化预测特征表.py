# -*- coding: utf-8 -*-
"""
由 03 距离场 + 剖面金矿体多边形 + 钻孔 Au 打标签：
  正样本：剖面金矿体多边形内部体素 + 钻孔 Au ≥ 边界品位
  硬负样本：钻孔 Au < 边界品位
  其余：随机负样本池（label=-1）
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree

from profile_real_coords import assign_profile_from_real, drop_excluded_sections, load_section_params
from ml_mineral_train import exclude_gold_from_candidates

root_dir = Path(__file__).resolve().parent

in_csv = root_dir / "三维地质概率场结果" / "03_三维地质距离场.csv"
section_param_csv = root_dir / "三维地质概率场结果" / "11_真实坐标结果" / "11_剖面真实坐标参数表.csv"
drill_xlsx = root_dir / "钻孔数据库_自动解析.xlsx"
parsed_excel = root_dir / "综合三维地质建模结果" / "01_明码文件综合解析结果.xlsx"

out_csv = root_dir / "三维地质概率场结果" / "04C_地表DEM约束矿化预测特征表_10m厚度版.csv"
out_summary = root_dir / "三维地质概率场结果" / "04C_厚度版标签统计.csv"
out_assay_map = root_dir / "三维地质概率场结果" / "04C_钻孔硬负样本映射.csv"
out_pos_map = root_dir / "三维地质概率场结果" / "04C_钻孔正样本映射.csv"
out_ore_map = root_dir / "三维地质概率场结果" / "04C_金矿体多边形正样本统计.csv"

# 边界品位（ppm，数值同 g·t⁻¹）：Au≥为正，其余化验全部为硬负
cutoff_au = 0.3
positive_sample_radius = 12.0
negative_sample_radius = 0.0  # 硬负：只落到最近体素，不沿孔扩邻域
sample_step = 5.0


def drill_point(collar_x, collar_y, collar_z, azimuth, dip, depth):
    az = math.radians(float(azimuth))
    dip_rad = math.radians(float(dip))
    horizontal = depth * math.cos(dip_rad)
    dz = depth * math.sin(dip_rad)
    x = collar_x + horizontal * math.sin(az)
    y = collar_y + horizontal * math.cos(az)
    z = collar_z + dz
    return x, y, z


def sample_interval_depths(start, end, step):
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
    for path in (drill_xlsx, parsed_excel):
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


def build_drill_assay_points(
    collar: pd.DataFrame,
    assay: pd.DataFrame,
    *,
    au_min: float | None = None,
    au_gt: float | None = None,
    au_max: float | None = None,
    au_lt: float | None = None,
) -> pd.DataFrame:
    need_c = ["HoleID", "X", "Y", "Z", "FinalAzimuth", "FinalDip"]
    need_a = ["HoleID", "From", "To", "Au_value"]
    if collar.empty or assay.empty:
        return pd.DataFrame()
    for c in need_c + need_a:
        cols = collar.columns if c in need_c else assay.columns
        if c not in cols:
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
    if au_min is not None:
        assay = assay[assay["Au_value"] >= au_min]
    if au_gt is not None:
        assay = assay[assay["Au_value"] > au_gt]
    if au_max is not None:
        assay = assay[assay["Au_value"] <= au_max]
    if au_lt is not None:
        assay = assay[assay["Au_value"] < au_lt]

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
        for d in sample_interval_depths(r.From, r.To, sample_step):
            px, py, pz = drill_point(h["X"], h["Y"], h["Z"], az, dip, d)
            rows.append({
                "HoleID": hid,
                "Au_value": float(r.Au_value),
                "sample_RealX": px,
                "sample_RealY": py,
                "sample_RealZ": pz,
            })
    return pd.DataFrame(rows)


def map_drill_points_to_voxels(
    df: pd.DataFrame,
    pts: pd.DataFrame,
    radius: float,
    label_source: str,
) -> tuple[pd.Index, pd.DataFrame]:
    if pts.empty:
        return pd.Index([], dtype=int), pd.DataFrame()

    grid_xyz = df[["RealX", "RealY", "RealZ"]].values.astype(float)
    tree = cKDTree(grid_xyz)
    mapped_rows = []
    hit = set()

    for r in pts.itertuples():
        p = np.array([r.sample_RealX, r.sample_RealY, r.sample_RealZ], dtype=float)
        if radius > 0:
            ids = tree.query_ball_point(p, r=radius)
            if not ids:
                _, idx = tree.query(p, k=1)
                ids = [int(idx)]
        else:
            _, idx = tree.query(p, k=1)
            ids = [int(idx)]
        for vid in ids:
            hit.add(int(vid))
            mapped_rows.append({
                "voxel_index": int(vid),
                "label_source": label_source,
                "HoleID": r.HoleID,
                "Au_value": r.Au_value,
                "RealX": float(df.iloc[vid]["RealX"]),
                "RealY": float(df.iloc[vid]["RealY"]),
                "RealZ": float(df.iloc[vid]["RealZ"]),
            })

    return pd.Index(sorted(hit), dtype=int), pd.DataFrame(mapped_rows)


def load_ore_polygons_by_section(excel_path: Path) -> dict[int, list[np.ndarray]]:
    """读取剖面金矿体 WAP 多边形（剖面局部 X-Z）。"""
    if not excel_path.exists():
        return {}

    pp = pd.read_excel(excel_path, sheet_name="polygon_points_wap")
    pp.columns = [str(c).strip() for c in pp.columns]
    pp = drop_excluded_sections(pp)
    pp = pp[pp["feature_type"].astype(str) == "金矿体"].copy()
    if pp.empty:
        return {}

    pp["section_id"] = pd.to_numeric(pp["section_id"], errors="coerce")
    pp = pp.dropna(subset=["section_id"])
    pp["section_id"] = pp["section_id"].astype(int)

    x_col = "model_x" if "model_x" in pp.columns else "x"
    z_col = "model_z" if "model_z" in pp.columns else "y"
    pp[x_col] = pd.to_numeric(pp[x_col], errors="coerce")
    pp[z_col] = pd.to_numeric(pp[z_col], errors="coerce")
    pp = pp.dropna(subset=[x_col, z_col, "local_polygon_order", "point_order"])

    out: dict[int, list[np.ndarray]] = {}
    group_cols = ["section_id", "source_file", "local_polygon_order"]
    for key, g in pp.groupby(group_cols):
        sid = int(key[0])
        g = g.sort_values("point_order")
        pts = g[[x_col, z_col]].values.astype(float)
        if len(pts) < 3:
            continue
        area = 0.5 * abs(
            np.dot(pts[:, 0], np.roll(pts[:, 1], -1))
            - np.dot(pts[:, 1], np.roll(pts[:, 0], -1))
        )
        if area <= 0:
            continue
        out.setdefault(sid, []).append(pts)
    return out


def mark_ore_polygon_positives(df: pd.DataFrame, polygons: dict[int, list[np.ndarray]]) -> tuple[pd.Index, pd.DataFrame]:
    """体素落在剖面金矿体多边形内 → 正样本。"""
    if not polygons:
        return pd.Index([], dtype=int), pd.DataFrame()

    ok = df["coord_status"].astype(str) == "ok"
    stats = []
    hit = np.zeros(len(df), dtype=bool)

    for sid, polys in sorted(polygons.items()):
        sec_mask = ok & (pd.to_numeric(df["section_id"], errors="coerce") == sid)
        pos = np.flatnonzero(sec_mask.to_numpy())
        if len(pos) == 0:
            continue

        xz = df.iloc[pos][["X", "Z"]].values.astype(float)
        inside = np.zeros(len(xz), dtype=bool)
        for poly in polys:
            inside |= MplPath(poly).contains_points(xz)

        hit[pos[inside]] = True
        stats.append({
            "section_id": sid,
            "polygon_count": len(polys),
            "section_voxels": len(pos),
            "positive_voxels": int(inside.sum()),
        })

    pos_idx = df.index[hit]
    return pos_idx, pd.DataFrame(stats)


if not in_csv.exists():
    raise FileNotFoundError(f"缺少输入：{in_csv}\n请先运行：python 3_三维地质距离场.py")

df = pd.read_csv(in_csv, encoding="utf-8-sig")
df.columns = [str(c).strip() for c in df.columns]

need = ["RealX", "RealY", "RealZ"]
missing = [c for c in need if c not in df.columns]
if missing:
    raise ValueError(f"缺少字段：{missing}（请先运行新版 2/3）")

print("输入体素数：", len(df))

params = load_section_params(section_param_csv)
df = assign_profile_from_real(df, params)
n_matched = int((df["coord_status"] == "ok").sum())
print(f"剖面匹配：{n_matched} / {len(df)}")

df["label"] = -1
df["label_source"] = "unlabeled"

# --- 正样本 1：剖面金矿体多边形内部 ---
ore_polys = load_ore_polygons_by_section(parsed_excel)
print(f"剖面金矿体：{sum(len(v) for v in ore_polys.values())} 个多边形，涉及 {len(ore_polys)} 条剖面")

ore_idx, ore_stats = mark_ore_polygon_positives(df, ore_polys)
if len(ore_idx):
    df.loc[ore_idx, "label"] = 1
    df.loc[ore_idx, "label_source"] = "ore_polygon"
    ore_stats.to_csv(out_ore_map, index=False, encoding="utf-8-sig")
    print(f"金矿体多边形正样本：{len(ore_idx)} 个体素")
    print(ore_stats.to_string(index=False))
else:
    print("[警告] 未匹配到金矿体多边形内部体素，请检查 WAP 与剖面坐标。")
    pd.DataFrame().to_csv(out_ore_map, index=False, encoding="utf-8-sig")

# --- 正样本 2：钻孔 Au ≥ 边界品位 ---
collar, assay = read_drill_tables()
pos_pts = build_drill_assay_points(collar, assay, au_min=cutoff_au)
pos_idx, pos_mapped = map_drill_points_to_voxels(
    df, pos_pts, positive_sample_radius, "drill_au_positive"
)
if len(pos_idx):
    new_pos = pos_idx[~pos_idx.isin(ore_idx)] if len(ore_idx) else pos_idx
    df.loc[pos_idx, "label"] = 1
    df.loc[new_pos, "label_source"] = "drill_au_positive"
    overlap = pos_idx.intersection(ore_idx) if len(ore_idx) else pd.Index([])
    if len(overlap):
        df.loc[overlap, "label_source"] = "ore_polygon+drill_au"
    print(f"钻孔 Au 正样本：{len(pos_pts)} 个样品点 → {len(pos_idx)} 个体素（Au≥{cutoff_au}）")
    if not pos_mapped.empty:
        pos_mapped.to_csv(out_pos_map, index=False, encoding="utf-8-sig")

# --- 硬负样本：钻孔 Au < 边界品位 ---
neg_pts = build_drill_assay_points(collar, assay, au_lt=cutoff_au)
hard_idx, hard_mapped = map_drill_points_to_voxels(
    df, neg_pts, negative_sample_radius, "drill_hard_negative"
)
if len(hard_idx):
    hard_cands = df.loc[hard_idx].copy()
    hard_cands = exclude_gold_from_candidates(df, hard_cands)
    can_neg = hard_cands.index
    df.loc[can_neg, "label"] = 0
    df.loc[can_neg, "label_source"] = "drill_hard_negative"
    hard_mapped = hard_mapped[hard_mapped["voxel_index"].isin(can_neg)] if not hard_mapped.empty else hard_mapped
    hard_mapped.to_csv(out_assay_map, index=False, encoding="utf-8-sig")
    print(f"钻孔硬负样本：{len(neg_pts)} 个样品点 → {len(can_neg)} 个体素（Au<{cutoff_au}，最近体素）")
else:
    print("[提示] 未生成钻孔硬负样本。")
    pd.DataFrame().to_csv(out_assay_map, index=False, encoding="utf-8-sig")

df.to_csv(out_csv, index=False, encoding="utf-8-sig")

summary = pd.DataFrame([
    {"item": "input_file", "value": str(in_csv)},
    {"item": "output_file", "value": str(out_csv)},
    {"item": "voxel_count", "value": len(df)},
    {"item": "section_matched", "value": n_matched},
    {"item": "cutoff_au", "value": cutoff_au},
    {"item": "positive_rule", "value": f"Au>={cutoff_au}"},
    {"item": "negative_rule", "value": f"Au<{cutoff_au}"},
    {"item": "negative_sample_radius", "value": negative_sample_radius},
    {"item": "ore_polygon_count", "value": sum(len(v) for v in ore_polys.values())},
    {"item": "positive_ore_polygon", "value": int((df["label_source"] == "ore_polygon").sum())},
    {"item": "positive_drill_au", "value": int((df["label_source"] == "drill_au_positive").sum())},
    {"item": "positive_total", "value": int((df["label"] == 1).sum())},
    {"item": "hard_negative_count", "value": int((df["label_source"] == "drill_hard_negative").sum())},
    {"item": "random_negative_pool", "value": int((df["label"] == -1).sum())},
])
summary.to_csv(out_summary, index=False, encoding="utf-8-sig")

print("\n完成。")
print("特征表（含标签）：", out_csv)
print("标签统计：\n", df["label"].value_counts())
print("\n标签来源：\n", df["label_source"].value_counts())

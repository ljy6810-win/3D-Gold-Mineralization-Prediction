# -*- coding: utf-8 -*-
"""矿化预测训练共用逻辑（钻孔级验证划分 + 最终全量模型）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from borehole_split import (  # noqa: F401
    HOLE_MAP_CSV,
    HOLE_SPLIT_CSV,
    borehole_masks_from_label,
    borehole_train_val_masks,
    borehole_train_val_test_masks,
    ensure_labeled_hole_map,
)
from profile_real_coords import load_section_params

HERE = Path(__file__).resolve().parent


def _resolve_root() -> Path:
    return HERE


ROOT_DIR = _resolve_root()

FEATURE_CSV = ROOT_DIR / "三维地质概率场结果" / "04C_地表DEM约束矿化预测特征表_10m厚度版.csv"
SECTION_PARAM_CSV = (
    ROOT_DIR / "三维地质概率场结果" / "11_真实坐标结果" / "11_剖面真实坐标参数表.csv"
)

DEFAULT_NEGATIVE_RATIO = 3
DEFAULT_RANDOM_STATE = 42
DEFAULT_PROB_THRESHOLD = 0.5
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_VIZ_PROB_FLOOR = 0.05
DEFAULT_MAX_VOXELS = 30000
DEFAULT_GRID_DXY = 10.0
DEFAULT_GRID_DZ = 10.0
DEFAULT_PUB_DPI = 300
# 可视化：格距 10 m；抽稀时按 stride 放大方块以贴合相邻格，略重叠消缝
VOXEL_DISPLAY_OVERLAP = 1.08
# 南北标注相对模型外扩（截图更干净）；东西稍近
NESW_NS_PAD_FRAC = 0.32
NESW_EW_PAD_FRAC = 0.12


def voxel_half_m(
    grid_dxy: float = DEFAULT_GRID_DXY,
    grid_dz: float = DEFAULT_GRID_DZ,
    overlap: float = VOXEL_DISPLAY_OVERLAP,
    stride: int = 1,
) -> tuple[float, float, float]:
    """
    正方体半边长（三轴相同）。
    预测网格为 10 m；抽稀 stride>1 时三轴同步放大，贴合且保持正方体。
    """
    s = max(1, int(stride))
    # 强制用水平格距，保证 hx=hy=hz（正方体，而非长方体）
    h = float(grid_dxy) / 2.0 * float(overlap) * s
    return (h, h, h)


def set_cubic_voxel_aspect(ax, mask_c: np.ndarray) -> None:
    """matplotlib voxels 默认会按轴长拉伸；按索引数设 box_aspect 才能看出正方体。"""
    nx, ny, nz = mask_c.shape
    ax.set_box_aspect((max(nx, 1), max(ny, 1), max(nz, 1)))


def map_view_camera() -> dict:
    """从南看向北、Z 朝上；配合 Y 轴反转 → 上北下南左西右东。"""
    return dict(
        eye=dict(x=-1.85, y=0.0, z=0.95),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
    )


def apply_map_nesw_screen(fig, y_min: float, y_max: float) -> None:
    """Plotly 默认图面：上北下南、左西右东。"""
    fig.update_layout(scene_yaxis_range=[y_max, y_min], scene_camera=map_view_camera())


def add_nesw_annotations_plotly(fig, df, *, ns_frac: float = NESW_NS_PAD_FRAC, ew_frac: float = NESW_EW_PAD_FRAC) -> None:
    """模型四周标方位；南北外扩更大，便于截到干净主体。"""
    x0, x1 = float(df["RealX"].min()), float(df["RealX"].max())
    y0, y1 = float(df["RealY"].min()), float(df["RealY"].max())
    z0, z1 = float(df["RealZ"].min()), float(df["RealZ"].max())
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    z_lab = z1 + max(40.0, (z1 - z0) * 0.08)
    fig.update_layout(
        scene=dict(
            annotations=[
                dict(x=x1 + ns_frac * dx, y=cy, z=z_lab, text="<b>北 N</b>", showarrow=False, font=dict(size=18, color="#111")),
                dict(x=cx, y=y1 + ew_frac * dy, z=z_lab, text="<b>东 E</b>", showarrow=False, font=dict(size=18, color="#111")),
                dict(x=cx, y=y0 - ew_frac * dy, z=z_lab, text="<b>西 W</b>", showarrow=False, font=dict(size=18, color="#111")),
            ]
        )
    )


def add_nesw_labels_matplotlib(ax, mask_c: np.ndarray, *, ns_frac: float = NESW_NS_PAD_FRAC, ew_frac: float = NESW_EW_PAD_FRAC) -> None:
    """体素索引空间：+X北 +Y东；南北拉远。"""
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    nx, ny, nz = mask_c.shape
    z_top = nz + max(4.0, nz * 0.10)
    cx, cy = nx / 2.0, ny / 2.0
    pad_ns = max(nx, ny) * ns_frac
    pad_ew = max(nx, ny) * ew_frac
    style = dict(fontsize=15, fontweight="bold", color="#111111", ha="center", va="center")
    ax.text(nx + pad_ns, cy, z_top, "北 N", **style)
    ax.text(cx, ny + pad_ew, z_top, "东 E", **style)
    ax.text(cx, -pad_ew, z_top, "西 W", **style)


def add_compass_inset(fig) -> None:
    """左下角指北针：上北下南左西右东。"""
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    axc = fig.add_axes([0.03, 0.05, 0.13, 0.13])
    axc.set_xlim(-1.2, 1.2)
    axc.set_ylim(-1.2, 1.2)
    axc.set_aspect("equal")
    axc.axis("off")
    axc.set_facecolor("none")
    axc.annotate("", xy=(0, 0.85), xytext=(0, -0.85), arrowprops=dict(arrowstyle="<->", color="#222", lw=1.6))
    axc.annotate("", xy=(0.85, 0), xytext=(-0.85, 0), arrowprops=dict(arrowstyle="<->", color="#222", lw=1.6))
    axc.annotate("", xy=(0, 1.05), xytext=(0, 0.55), arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=2.2, mutation_scale=12))
    kw = dict(ha="center", va="center", fontsize=11, fontweight="bold", color="#111")
    axc.text(0, 1.15, "北 N", **kw)
    axc.text(1.15, 0, "东 E", **kw)
    axc.text(-1.15, 0, "西 W", **kw)


def add_nesw_labels_matplotlib_xyz(ax, bounds: dict, *, ns_frac: float = NESW_NS_PAD_FRAC, ew_frac: float = NESW_EW_PAD_FRAC) -> None:
    """工程坐标空间标注方位（非体素索引）。"""
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    x0, x1 = float(bounds["xmin"]), float(bounds["xmax"])
    y0, y1 = float(bounds["ymin"]), float(bounds["ymax"])
    z0, z1 = float(bounds["zmin"]), float(bounds["zmax"])
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    z_lab = z1 + max(40.0, (z1 - z0) * 0.08)
    style = dict(fontsize=15, fontweight="bold", color="#111111", ha="center", va="center")
    ax.text(x1 + ns_frac * dx, cy, z_lab, "北 N", **style)
    ax.text(cx, y1 + ew_frac * dy, z_lab, "东 E", **style)
    ax.text(cx, y0 - ew_frac * dy, z_lab, "西 W", **style)


def apply_map_matplotlib_view(ax) -> None:
    """论文图默认视角：上北下南左西右东。"""
    ax.view_init(elev=25, azim=180)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y1, y0)


# 视为「已确定金矿」的正样本来源
POSITIVE_LABEL_SOURCES = frozenset({
    "ore_polygon",
    "drill_au_positive",
    "ore_polygon+drill_au",
})


def gold_positive_mask(df: pd.DataFrame) -> pd.Series:
    """已确定金矿体素：label=1 或正样本 label_source。"""
    mask = pd.to_numeric(df["label"], errors="coerce") == 1
    if "label_source" in df.columns:
        mask |= df["label_source"].astype(str).isin(POSITIVE_LABEL_SOURCES)
    return mask


def exclude_gold_from_candidates(
    df: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """从负样本候选里去掉金矿体素（同 index 或同 RealX/Y/Z 格点）。"""
    if candidates.empty:
        return candidates

    out = candidates[~candidates.index.isin(df.index[gold_positive_mask(df)])].copy()
    if out.empty:
        return out

    if not {"RealX", "RealY", "RealZ"}.issubset(df.columns):
        return out

    pos = df.loc[gold_positive_mask(df), ["RealX", "RealY", "RealZ"]]
    pos_keys = {
        (round(float(r.RealX), 3), round(float(r.RealY), 3), round(float(r.RealZ), 3))
        for r in pos.itertuples()
    }
    if not pos_keys:
        return out

    cand_keys = [
        (round(float(r.RealX), 3), round(float(r.RealY), 3), round(float(r.RealZ), 3))
        for r in out.itertuples()
    ]
    keep = [k not in pos_keys for k in cand_keys]
    return out.loc[out.index[keep]].copy()


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    prob_cols = [c for c in df.columns if c.startswith("P_")]
    cols = prob_cols + [
        "fault_dist",
        "alter_dist",
        "structure_dist",
        "contact_dist",
        "DepthBelowSurface",
        "relative_depth_norm",
        "SurfaceElev",
        "InSurfaceHull",
        "DEM_match_dist_norm",
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少训练特征：{missing}")
    return cols


def get_section_group(df: pd.DataFrame) -> pd.Series:
    if "section_id" in df.columns and df["section_id"].notna().sum() > 0:
        return pd.to_numeric(df["section_id"], errors="coerce")
    if "S" in df.columns:
        return pd.to_numeric(df["S"], errors="coerce")
    raise ValueError("特征表需要 section_id 或 S 才能做剖面留一验证。")


def build_training_set(
    df: pd.DataFrame,
    train_mask: pd.Series,
    pos_train: pd.DataFrame,
    fold_seed: int,
    *,
    negative_ratio: int = DEFAULT_NEGATIVE_RATIO,
) -> pd.DataFrame | None:
    if pos_train.empty:
        return None

    hard_neg = df[
        train_mask
        & (df["label"] == 0)
        & (df.get("label_source", pd.Series("", index=df.index)).astype(str) == "drill_hard_negative")
    ].copy()
    hard_neg = exclude_gold_from_candidates(df, hard_neg)

    random_pool = df[train_mask & (df["label"] == -1)]
    random_pool = exclude_gold_from_candidates(df, random_pool)

    n_neg_target = len(pos_train) * negative_ratio

    parts = [pos_train]
    n_hard = min(len(hard_neg), n_neg_target)
    if n_hard:
        parts.append(
            hard_neg.sample(n=n_hard, random_state=fold_seed)
            if n_hard < len(hard_neg)
            else hard_neg
        )

    n_random = n_neg_target - n_hard
    if n_random > 0 and len(random_pool):
        n_random = min(n_random, len(random_pool))
        rnd = random_pool.sample(n=n_random, random_state=fold_seed + 1).copy()
        rnd["label"] = 0
        if "label_source" in rnd.columns:
            rnd["label_source"] = "random_negative_train"
        parts.append(rnd)

    train_df = pd.concat(parts, ignore_index=False)
    # 最终再滤一遍，确保负类行中无金矿体素
    if (train_df["label"] == 0).any():
        neg_part = train_df[train_df["label"] == 0]
        neg_part = exclude_gold_from_candidates(df, neg_part)
        train_df = pd.concat([train_df[train_df["label"] == 1], neg_part], ignore_index=False)
    if (train_df["label"] == 0).sum() == 0:
        return None
    return train_df


def count_train_negatives(train_df: pd.DataFrame) -> tuple[int, int, int]:
    n_train_neg = int((train_df["label"] == 0).sum())
    if "label_source" in train_df.columns:
        n_hard = int((train_df["label_source"] == "drill_hard_negative").sum())
    else:
        n_hard = n_train_neg
    return int((train_df["label"] == 1).sum()), n_hard, n_train_neg - n_hard


def feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values


def load_mineral_table(
    feature_csv: Path | None = None,
    section_param_csv: Path | None = None,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    feature_csv = feature_csv or FEATURE_CSV
    section_param_csv = section_param_csv or SECTION_PARAM_CSV

    df = pd.read_csv(feature_csv, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    params = load_section_params(section_param_csv)

    if "label" not in df.columns:
        raise ValueError("特征表缺少 label 字段。")
    if not {"RealX", "RealY", "RealZ"}.issubset(df.columns):
        raise ValueError("特征表缺少 RealX / RealY / RealZ。")

    df["section_group"] = get_section_group(df)
    feature_cols = get_feature_cols(df)
    return df, feature_cols, params


def positive_section_ids(df: pd.DataFrame) -> list[float]:
    return sorted(df.loc[df["label"] == 1, "section_group"].dropna().unique())


def random_train_val_masks(
    df: pd.DataFrame,
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.Series, pd.Series]:
    """已废弃的体素随机划分，仅保留对照。论文评价请用 borehole_train_val_masks。"""
    labeled_idx = df.index[df["label"].isin([0, 1])]
    if len(labeled_idx) < 10:
        raise ValueError("已标注样本过少，无法划分验证集。")

    y = df.loc[labeled_idx, "label"].astype(int)
    if y.nunique() < 2:
        raise ValueError("已标注样本缺少正负两类，无法划分验证集。")

    train_idx, val_idx = train_test_split(
        labeled_idx,
        test_size=val_fraction,
        random_state=seed,
        stratify=y,
    )
    val_mask = df.index.isin(val_idx)
    train_mask = ~val_mask
    return train_mask, val_mask


def mineral_prob_colorscale() -> list[list]:
    """低概率→高概率连续渐变：深蓝 → 青 → 黄 → 橙 → 深红（印刷友好）。"""
    return [
        [0.0, "#1a237e"],
        [0.20, "#1565c0"],
        [0.40, "#4fc3f7"],
        [0.55, "#ffeb3b"],
        [0.70, "#ff9800"],
        [0.85, "#e65100"],
        [1.0, "#b71c1c"],
    ]


def _matplotlib_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "mineral_prob",
        [(p, c) for p, c in mineral_prob_colorscale()],
    )


def probability_volume_grid(
    df: pd.DataFrame,
    prob_col: str,
    *,
    grid_dxy: float = DEFAULT_GRID_DXY,
    grid_dz: float = DEFAULT_GRID_DZ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """将预测概率映射到规则工程网格体（同格点取最大概率）。"""
    work = df[["RealX", "RealY", "RealZ", prob_col]].copy()
    work["gx"] = (np.round(work["RealX"].astype(float) / grid_dxy) * grid_dxy).astype(np.float64)
    work["gy"] = (np.round(work["RealY"].astype(float) / grid_dxy) * grid_dxy).astype(np.float64)
    work["gz"] = (np.round(work["RealZ"].astype(float) / grid_dz) * grid_dz).astype(np.float64)
    work["prob_val"] = work[prob_col].astype(float)

    agg = work.groupby(["gx", "gy", "gz"], as_index=False)["prob_val"].max()
    x_vals = np.sort(agg["gx"].unique())
    y_vals = np.sort(agg["gy"].unique())
    z_vals = np.sort(agg["gz"].unique())

    x_to_i = {v: i for i, v in enumerate(x_vals)}
    y_to_i = {v: i for i, v in enumerate(y_vals)}
    z_to_i = {v: i for i, v in enumerate(z_vals)}

    volume = np.zeros((len(x_vals), len(y_vals), len(z_vals)), dtype=np.float32)
    for row in agg.itertuples():
        volume[x_to_i[row.gx], y_to_i[row.gy], z_to_i[row.gz]] = row.prob_val

    return x_vals, y_vals, z_vals, volume


def _select_voxel_indices(
    volume: np.ndarray,
    *,
    prob_floor: float,
    max_voxels: int,
    seed: int,
) -> np.ndarray:
    """返回待渲染体素的三维索引 (N, 3)。"""
    mask = volume >= prob_floor
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return idx

    if len(idx) <= max_voxels:
        return idx

    probs = volume[mask]
    order = np.argsort(-probs)
    top = order[:max_voxels]
    return idx[top]


# 单位立方体 12 三角面（顶点 0–7）
_CUBE_I = np.array([0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
_CUBE_J = np.array([1, 2, 5, 6, 4, 7, 5, 6, 3, 7, 0, 7], dtype=np.int32)
_CUBE_K = np.array([2, 3, 6, 7, 5, 3, 6, 2, 7, 4, 4, 6], dtype=np.int32)


def _batched_voxel_mesh(
    centers: np.ndarray,
    half: float | tuple[float, float, float] | list[float],
    probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """批量构建体素立方体 Mesh3d 顶点与面。half 可为标量或 (hx, hy, hz)。"""
    n = len(centers)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        empty_i = np.array([], dtype=np.int32)
        return empty, empty, empty, empty_i, empty_i, empty_i, empty

    if np.isscalar(half):
        half_xyz = np.array([float(half), float(half), float(half)], dtype=np.float64)
    else:
        half_xyz = np.asarray(half, dtype=np.float64).reshape(3)

    # 8 顶点 / 立方体
    local = np.array(
        [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    verts = centers[:, None, :] + local[None, :, :] * half_xyz[None, None, :]
    verts = verts.reshape(-1, 3)
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]

    offsets = (np.arange(n, dtype=np.int32) * 8)[:, None]
    tri_i = (_CUBE_I[None, :] + offsets).reshape(-1)
    tri_j = (_CUBE_J[None, :] + offsets).reshape(-1)
    tri_k = (_CUBE_K[None, :] + offsets).reshape(-1)

    intensity = np.repeat(probs.astype(np.float64), 8)
    return x, y, z, tri_i, tri_j, tri_k, intensity


def _crop_volume_bbox(
    volume: np.ndarray,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    z_vals: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """裁剪到有效体素包围盒，便于 matplotlib voxels 渲染。"""
    if not mask.any():
        return x_vals[:0], y_vals[:0], z_vals[:0], np.zeros((0, 0, 0), dtype=np.float32)

    xi = np.where(mask.any(axis=(1, 2)))[0]
    yi = np.where(mask.any(axis=(0, 2)))[0]
    zi = np.where(mask.any(axis=(0, 1)))[0]
    sl = (slice(xi[0], xi[-1] + 1), slice(yi[0], yi[-1] + 1), slice(zi[0], zi[-1] + 1))
    return x_vals[sl[0]], y_vals[sl[1]], z_vals[sl[2]], volume[sl], mask[sl]


def build_publication_voxel_viz(
    df: pd.DataFrame,
    prob_col: str,
    out_html: Path | str,
    *,
    title: str,
    model_label: str = "矿化概率",
    prob_threshold: float = DEFAULT_PROB_THRESHOLD,
    prob_floor: float = DEFAULT_VIZ_PROB_FLOOR,
    max_voxels: int = DEFAULT_MAX_VOXELS,
    grid_dxy: float = DEFAULT_GRID_DXY,
    grid_dz: float = DEFAULT_GRID_DZ,
    seed: int = DEFAULT_RANDOM_STATE,
    auto_open: bool = False,
    dpi: int = DEFAULT_PUB_DPI,
) -> None:
    """论文级体素方块可视化：交互 HTML + 高分辨率 PNG/PDF。"""
    import plotly.graph_objects as go
    import plotly.io as pio

    if not {"RealX", "RealY", "RealZ"}.issubset(df.columns):
        raise ValueError("可视化需要 RealX / RealY / RealZ 列。")

    out_html = Path(out_html)
    out_png = Path(str(out_html).replace(".html", "_论文图.png"))
    out_pdf = Path(str(out_html).replace(".html", "_论文图.pdf"))

    x_vals, y_vals, z_vals, volume = probability_volume_grid(
        df, prob_col, grid_dxy=grid_dxy, grid_dz=grid_dz
    )
    idx = _select_voxel_indices(volume, prob_floor=prob_floor, max_voxels=max_voxels, seed=seed)
    if len(idx) == 0:
        raise ValueError(f"没有 P≥{prob_floor} 的体素可渲染，请降低 prob_floor。")

    centers = np.column_stack([
        x_vals[idx[:, 0]],
        y_vals[idx[:, 1]],
        z_vals[idx[:, 2]],
    ])
    probs = volume[idx[:, 0], idx[:, 1], idx[:, 2]]
    half = voxel_half_m(grid_dxy, grid_dz)
    mesh_x, mesh_y, mesh_z, tri_i, tri_j, tri_k, intensity = _batched_voxel_mesh(
        centers, half=half, probs=probs
    )

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=mesh_x,
            y=mesh_y,
            z=mesh_z,
            i=tri_i,
            j=tri_j,
            k=tri_k,
            intensity=intensity,
            colorscale=mineral_prob_colorscale(),
            cmin=0.0,
            cmax=1.0,
            flatshading=True,
            lighting=dict(ambient=0.55, diffuse=0.85, specular=0.15, roughness=0.9),
            lightposition=dict(x=0, y=0, z=100000),
            opacity=1.0,
            name=f"体素场 P≥{prob_floor}",
            showscale=True,
            showlegend=False,
            colorbar=dict(
                title=dict(text=f"{model_label}<br>(0→1)", side="right", font=dict(size=18)),
                tickmode="linear",
                tick0=0.0,
                dtick=0.2,
                len=0.55,
                thickness=24,
                outlinewidth=0,
                x=1.02,
                y=0.5,
                xanchor="left",
                yanchor="middle",
                xpad=8,
                tickfont=dict(size=18),
            ),
            hovertemplate="X=%{x:.1f} m<br>Y=%{y:.1f} m<br>Z=%{z:.1f} m<br>P=%{intensity:.3f}<extra></extra>",
        )
    )

    if "label" in df.columns:
        pos = df[df["label"] == 1]
        if not pos.empty:
            px, py, pz, pi, pj, pk, pint = _batched_voxel_mesh(
                pos[["RealX", "RealY", "RealZ"]].astype(float).values,
                half=tuple(0.95 * h for h in half),
                probs=np.ones(len(pos)),
            )
            fig.add_trace(
                go.Mesh3d(
                    x=px, y=py, z=pz, i=pi, j=pj, k=pk,
                    color="#00c853",
                    opacity=0.35,
                    flatshading=True,
                    name="已知矿体",
                    showscale=False,
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

    blank = dict(
        title=dict(text=""),
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        showbackground=False,
        showspikes=False,
        ticks="",
    )
    n_vox = len(idx)
    fig.update_layout(
        title=dict(
            text=(
                f"{title}<br>"
                f"<sup>10 m 体素 | +X北 +Y东 | 图面左西右东 | "
                f"展示 {n_vox:,} 个体素 (P≥{prob_floor}) | 参考阈值 P={prob_threshold}</sup>"
            ),
            font=dict(size=22),
        ),
        scene=dict(
            xaxis=blank,
            yaxis=blank,
            zaxis=blank,
            aspectmode="data",
            bgcolor="white",
            camera=map_view_camera(),
        ),
        paper_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif", size=16, color="#222"),
        legend=dict(
            title=dict(text="图层", font=dict(size=18)),
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(size=16),
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
        ),
        width=1800,
        height=1000,
        margin=dict(l=40, r=140, t=90, b=40),
    )
    show_xy = pd.DataFrame({
        "RealX": centers[:, 0],
        "RealY": centers[:, 1],
        "RealZ": centers[:, 2],
    })
    add_nesw_annotations_plotly(fig, show_xy)
    apply_map_nesw_screen(fig, float(show_xy["RealY"].min()), float(show_xy["RealY"].max()))

    pio.write_html(fig, file=str(out_html), include_plotlyjs="cdn", auto_open=auto_open)

    # 高分辨率静态图（论文用）
    _export_matplotlib_voxel_figure(
        x_vals, y_vals, z_vals, volume,
        prob_floor=prob_floor,
        max_voxels=max_voxels,
        title=title,
        model_label=model_label,
        prob_threshold=prob_threshold,
        grid_dxy=grid_dxy,
        out_png=out_png,
        out_pdf=out_pdf,
        dpi=dpi,
    )


def _export_matplotlib_voxel_figure(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    z_vals: np.ndarray,
    volume: np.ndarray,
    *,
    prob_floor: float,
    max_voxels: int,
    title: str,
    model_label: str,
    prob_threshold: float,
    grid_dxy: float,
    out_png: Path,
    out_pdf: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    cmap = _matplotlib_cmap()
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    mask = volume >= prob_floor
    if mask.sum() > max_voxels:
        flat_idx = np.argwhere(mask)
        flat_probs = volume[mask]
        keep = np.argsort(-flat_probs)[:max_voxels]
        mask = np.zeros_like(mask, dtype=bool)
        for i in keep:
            mask[tuple(flat_idx[i])] = True

    cx, cy, cz, vol_c, mask_c = _crop_volume_bbox(volume, x_vals, y_vals, z_vals, mask)
    if vol_c.size == 0:
        return

    facecolors = np.zeros(vol_c.shape + (4,), dtype=np.float32)
    facecolors[..., :] = cmap(norm(vol_c))[..., :]
    facecolors[..., 3] = np.where(mask_c, 0.92, 0.0)

    fig = plt.figure(figsize=(12, 9), dpi=dpi, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(
        mask_c,
        facecolors=facecolors,
        edgecolor=(0.15, 0.15, 0.15, 0.08),
        linewidth=0.15,
        shade=True,
    )

    ax.set_axis_off()
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor("white")
        axis.pane.set_alpha(0)
    ax.grid(False)
    ax.set_title(
        f"{title}\n(10 m voxels, P≥{prob_floor})",
        fontsize=18,
        pad=16,
        color="#222222",
    )
    set_cubic_voxel_aspect(ax, mask_c)
    apply_map_matplotlib_view(ax)
    add_nesw_labels_matplotlib(ax, mask_c)
    add_compass_inset(fig)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02, fraction=0.035)
    cbar.set_label(model_label, fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    fig.subplots_adjust(left=0.06, right=0.88, top=0.90, bottom=0.06)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.35)
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white", pad_inches=0.35)
    plt.close(fig)


def build_probability_gradient_html(
    df: pd.DataFrame,
    prob_col: str,
    out_html: Path | str,
    **kwargs,
) -> None:
    """兼容旧接口：转发到论文级体素方块可视化。"""
    build_publication_voxel_viz(df, prob_col, out_html, **kwargs)

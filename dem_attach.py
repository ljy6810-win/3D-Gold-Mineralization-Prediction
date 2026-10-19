# -*- coding: utf-8 -*-
"""从 surface_dem_grid.csv 为体素/网格挂地表高程与埋深特征。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DEFAULT_DEM_CSV = Path(__file__).resolve().parent / "data" / "surface_dem_grid.csv"


def _dem_grid_step(dem: pd.DataFrame) -> float:
    xs = np.sort(dem["X"].dropna().unique())
    if len(xs) < 2:
        return 30.0
    diffs = np.diff(xs)
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if len(diffs) else 30.0


def attach_dem_from_grid(
    df: pd.DataFrame,
    dem_csv: Path | None = None,
    x_col: str = "RealX",
    y_col: str = "RealY",
    z_col: str = "RealZ",
) -> pd.DataFrame:
    """最近 DEM 格点赋值 SurfaceElev，并计算埋深相关列。"""
    path = Path(dem_csv or DEFAULT_DEM_CSV)
    if not path.exists():
        alt = path.parent.parent / path.name
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(
                f"找不到 DEM 网格：{path}\n请先运行：python 地表高程插值.py"
            )

    dem = pd.read_csv(path, encoding="utf-8-sig")
    dem.columns = [str(c).strip() for c in dem.columns]
    dem = dem.dropna(subset=["X", "Y", "SurfaceElev"]).copy()
    if dem.empty:
        raise ValueError(f"DEM 网格无有效高程：{path}")

    out = df.copy()
    x = pd.to_numeric(out[x_col], errors="coerce")
    y = pd.to_numeric(out[y_col], errors="coerce")
    z = pd.to_numeric(out[z_col], errors="coerce")
    valid = x.notna() & y.notna() & z.notna()
    if not valid.any():
        raise ValueError(f"没有有效的 {x_col}/{y_col}/{z_col} 可用于挂 DEM。")

    step = _dem_grid_step(dem)
    tree = cKDTree(dem[["X", "Y"]].values.astype(float))
    vxy = np.column_stack([x[valid].values, y[valid].values])
    dist, idx = tree.query(vxy, k=1)

    for col in [
        "SurfaceElev",
        "DEM_match_dist_m",
        "DEM_match_dist_norm",
        "InSurfaceHull",
        "DepthBelowSurface",
        "relative_depth_norm",
    ]:
        out[col] = np.nan

    out.loc[valid, "SurfaceElev"] = dem["SurfaceElev"].values[idx].astype(float)
    out.loc[valid, "DEM_match_dist_m"] = dist.astype(float)
    out.loc[valid, "DEM_match_dist_norm"] = dist.astype(float) / step
    if "InHull" in dem.columns:
        out.loc[valid, "InSurfaceHull"] = dem["InHull"].values[idx].astype(int)
    else:
        out.loc[valid, "InSurfaceHull"] = 1

    out.loc[valid, "DepthBelowSurface"] = (
        out.loc[valid, "SurfaceElev"] - z[valid]
    )
    dmax = float(out.loc[valid, "DepthBelowSurface"].max())
    out.loc[valid, "relative_depth_norm"] = (
        out.loc[valid, "DepthBelowSurface"] / dmax if dmax > 0 else 0.0
    )
    return out


def filter_below_surface(
    df: pd.DataFrame,
    z_col: str = "RealZ",
    tol_m: float = 5.0,
) -> pd.DataFrame:
    """去掉地表以上及 DEM 凸包外的体素。"""
    if "SurfaceElev" not in df.columns:
        return df
    z = pd.to_numeric(df[z_col], errors="coerce")
    mask = df["SurfaceElev"].notna() & z.notna() & (z <= df["SurfaceElev"] + tol_m)
    if "InSurfaceHull" in df.columns:
        mask &= df["InSurfaceHull"].astype(int) == 1
    return df.loc[mask].copy()

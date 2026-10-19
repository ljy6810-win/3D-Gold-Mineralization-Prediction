# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from dem_attach import attach_dem_from_grid, filter_below_surface

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "results"

DXY = 10.0
DZ = 10.0
SIGMA = 50.0
INFLUENCE_DISTANCE = 60.0
QUATERNARY_BUFFER = 10.0
SURFACE_TOL_M = 5.0
SURFACE_GEOLOGY_DEPTH_M = 100.0
SURFACE_XY_SIGMA = 100.0
SURFACE_XY_MAX_M = 250.0
SURFACE_FILL_MIN_PROB = 0.85
NEAR_SURFACE_PROB = 0.12


def load_lithology(path: Path) -> pd.DataFrame:
    lith = pd.read_csv(path, encoding="utf-8-sig")
    lith.columns = [str(c).strip() for c in lith.columns]
    for col in ("RealX", "RealY", "RealZ", "岩性"):
        if col not in lith.columns:
            raise ValueError(f"missing column: {col}")
    lith = lith.dropna(subset=["RealX", "RealY", "RealZ", "岩性"]).copy()
    lith["RealX"] = lith["RealX"].astype(float)
    lith["RealY"] = lith["RealY"].astype(float)
    lith["RealZ"] = lith["RealZ"].astype(float)
    lith["岩性"] = lith["岩性"].astype(str).str.strip()
    lith = lith[lith["岩性"].str.len() > 0].copy()
    if lith.empty:
        raise ValueError("no lithology samples")
    return lith


def grid_bounds(lith: pd.DataFrame, dem_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dem = pd.read_csv(dem_csv, encoding="utf-8-sig")
    dem.columns = [str(c).strip() for c in dem.columns]
    dem = dem.dropna(subset=["X", "Y"])
    if "InHull" in dem.columns:
        dem = dem[dem["InHull"].astype(int) == 1].copy()
    if dem.empty:
        raise ValueError(f"empty DEM hull: {dem_csv}")
    rx_min = min(np.floor(dem["X"].min() / DXY) * DXY, np.floor(lith["RealX"].min() / DXY) * DXY)
    rx_max = max(np.ceil(dem["X"].max() / DXY) * DXY, np.ceil(lith["RealX"].max() / DXY) * DXY)
    ry_min = min(np.floor(dem["Y"].min() / DXY) * DXY, np.floor(lith["RealY"].min() / DXY) * DXY)
    ry_max = max(np.ceil(dem["Y"].max() / DXY) * DXY, np.ceil(lith["RealY"].max() / DXY) * DXY)
    rz_min = np.floor(lith["RealZ"].min() / DZ) * DZ
    rz_max = np.ceil(lith["RealZ"].max() / DZ) * DZ
    return (
        np.arange(rx_min, rx_max + DXY, DXY),
        np.arange(ry_min, ry_max + DXY, DXY),
        np.arange(rz_min, rz_max + DZ, DZ),
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lith_csv = DATA / "lithology_samples.csv"
    dem_csv = DATA / "surface_dem_grid.csv"
    lith = load_lithology(lith_csv)
    names = sorted(lith["岩性"].unique())
    print("classes", len(names))
    print(lith["岩性"].value_counts().to_string())

    rx, ry, rz = grid_bounds(lith, dem_csv)
    xx, yy, zz = np.meshgrid(rx, ry, rz, indexing="ij")
    grid = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    print("voxels_before_dem", len(grid))

    grid_df = pd.DataFrame({"RealX": grid[:, 0], "RealY": grid[:, 1], "RealZ": grid[:, 2]})
    grid_df = attach_dem_from_grid(grid_df, dem_csv)
    grid_df = filter_below_surface(grid_df, tol_m=SURFACE_TOL_M)
    grid = grid_df[["RealX", "RealY", "RealZ"]].values.astype(float)
    print("voxels_after_dem", len(grid))

    q = lith[lith["岩性"] == "第四系"].copy()
    if q.empty:
        q_base = np.full(len(grid), np.nan)
    else:
        q_tree = cKDTree(q[["RealX", "RealY"]].values)
        _, q_idx = q_tree.query(grid[:, :2], k=1)
        q_base = q["RealZ"].values[q_idx]

    cols = []
    for name in names:
        print("class", name)
        pts = lith.loc[lith["岩性"] == name, ["RealX", "RealY", "RealZ"]].values
        if len(pts) == 0:
            prob = np.zeros(len(grid))
        else:
            dist, _ = cKDTree(pts).query(grid, k=1)
            prob = np.exp(-(dist ** 2) / (2.0 * SIGMA ** 2))
            prob[dist > INFLUENCE_DISTANCE] *= 0.2
        if name == "第四系":
            deep = (~np.isnan(q_base)) & (grid[:, 2] < (q_base - QUATERNARY_BUFFER))
            prob[deep] = 0.0
            prob[np.isnan(q_base)] = 0.0
        cols.append(prob)

    raw = np.column_stack(cols)
    total = raw.sum(axis=1)
    valid = total > 1e-12
    prob = np.zeros_like(raw)
    prob[valid] = raw[valid] / total[valid, None]
    max_idx = np.argmax(prob, axis=1)
    max_prob = np.max(prob, axis=1)
    hard = np.array(names, dtype=object)[max_idx]
    hard[max_prob <= 0] = "未知"

    if "data_source" in lith.columns:
        geo = lith[lith["data_source"].astype(str) == "GeologyMap"].copy()
    else:
        geo = pd.DataFrame()
    if not geo.empty and "DepthBelowSurface" in grid_df.columns:
        tree2d = cKDTree(geo[["RealX", "RealY"]].values.astype(float))
        geo_names = geo["岩性"].astype(str).values
        depth = grid_df["DepthBelowSurface"].values.astype(float)
        need = (depth <= SURFACE_GEOLOGY_DEPTH_M) & ((hard == "未知") | (max_prob < NEAR_SURFACE_PROB))
        if need.any():
            qi = np.where(need)[0]
            dist2d, nn = tree2d.query(grid[qi, :2], k=1)
            index = {name: i for i, name in enumerate(names)}
            filled = 0
            for k, vi in enumerate(qi):
                if float(dist2d[k]) > SURFACE_XY_MAX_M:
                    continue
                lit = str(geo_names[int(nn[k])])
                if lit not in index:
                    continue
                p_fill = max(
                    float(np.exp(-(float(dist2d[k]) ** 2) / (2.0 * SURFACE_XY_SIGMA ** 2))),
                    SURFACE_FILL_MIN_PROB,
                )
                hard[vi] = lit
                max_prob[vi] = p_fill
                prob[vi, :] = 0.0
                prob[vi, index[lit]] = 1.0
                filled += 1
            print("surface_fill", filled, int(need.sum()))

    out = pd.DataFrame({"RealX": grid[:, 0], "RealY": grid[:, 1], "RealZ": grid[:, 2]})
    for col in (
        "SurfaceElev",
        "DepthBelowSurface",
        "relative_depth_norm",
        "InSurfaceHull",
        "DEM_match_dist_norm",
    ):
        if col in grid_df.columns:
            out[col] = grid_df[col].values
    for i, name in enumerate(names):
        out[f"P_{name}"] = prob[:, i]
    out["max_probability"] = max_prob
    out["max_lithology"] = hard

    prob_csv = OUT / "lithology_probability.csv"
    class_csv = OUT / "lithology_class.csv"
    summary_csv = OUT / "lithology_probability_summary.csv"
    out.to_csv(prob_csv, index=False, encoding="utf-8-sig")
    out[["RealX", "RealY", "RealZ", "max_lithology", "max_probability"]].to_csv(
        class_csv, index=False, encoding="utf-8-sig"
    )
    unknown = out["max_lithology"] == "未知"
    near50 = int((unknown & (out["DepthBelowSurface"] <= 50)).sum()) if "DepthBelowSurface" in out.columns else -1
    summary = pd.DataFrame(
        [
            {"item": "voxel_count", "value": len(out)},
            {"item": "dxy_m", "value": DXY},
            {"item": "dz_m", "value": DZ},
            {"item": "sigma_m", "value": SIGMA},
            {"item": "influence_distance_m", "value": INFLUENCE_DISTANCE},
            {"item": "quaternary_buffer_m", "value": QUATERNARY_BUFFER},
            {"item": "surface_tol_m", "value": SURFACE_TOL_M},
            {"item": "surface_geology_depth_m", "value": SURFACE_GEOLOGY_DEPTH_M},
            {"item": "surface_xy_sigma_m", "value": SURFACE_XY_SIGMA},
            {"item": "surface_xy_max_m", "value": SURFACE_XY_MAX_M},
            {"item": "surface_fill_min_prob", "value": SURFACE_FILL_MIN_PROB},
            {"item": "q_sample_min_z", "value": float(q["RealZ"].min()) if not q.empty else np.nan},
            {"item": "q_sample_max_z", "value": float(q["RealZ"].max()) if not q.empty else np.nan},
            {"item": "unknown_voxels", "value": int(unknown.sum())},
            {"item": "unknown_near_surface_le50m", "value": near50},
            {"item": "real_x_min", "value": float(out["RealX"].min())},
            {"item": "real_x_max", "value": float(out["RealX"].max())},
            {"item": "real_y_min", "value": float(out["RealY"].min())},
            {"item": "real_y_max", "value": float(out["RealY"].max())},
        ]
    )
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(prob_csv)
    print(class_csv)
    print(summary_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

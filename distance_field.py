# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from profile_real_coords import convert_profile_to_real, drop_excluded_sections, get_section_id, load_section_params

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "results"

FAULT_DECAY = 600.0
ALTER_DECAY = 180.0
STRUCTURE_DECAY = 180.0
ORE_DECAY = 40.0
CONTACT_DECAY = 50.0
CONTACT_RADIUS = 12.0


def feature_points(df: pd.DataFrame, feature_type: str) -> np.ndarray:
    g = df[df["feature_type"] == feature_type]
    if g.empty:
        return np.empty((0, 3))
    return g[["RealX", "RealY", "RealZ"]].dropna().values


def contact_points(prob: pd.DataFrame, radius: float) -> np.ndarray:
    coords = prob[["RealX", "RealY", "RealZ"]].values
    liths = prob["max_lithology"].astype(str).values
    tree = cKDTree(coords)
    kept = []
    for i in range(len(prob)):
        if liths[i] == "未知":
            continue
        nbs = tree.query_ball_point(coords[i], r=radius)
        if any(liths[j] != liths[i] and liths[j] != "未知" for j in nbs if j != i):
            kept.append(i)
    if not kept:
        return np.empty((0, 3))
    return coords[kept]


def distance_to_points(grid: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts is None or len(pts) == 0:
        return np.full(len(grid), np.nan)
    dist, _ = cKDTree(pts).query(grid)
    return dist


def exp_factor(dist: np.ndarray, decay: float) -> np.ndarray:
    dist = np.asarray(dist, dtype=float)
    out = np.zeros_like(dist)
    mask = ~np.isnan(dist)
    out[mask] = np.exp(-dist[mask] / decay)
    return out


def read_sheet(path: Path, sheet: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(path, sheet_name=sheet)
    except Exception:
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def safe_stat(arr, func):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0 or np.all(np.isnan(arr)):
        return np.nan
    return func(arr)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    prob = pd.read_csv(OUT / "lithology_probability.csv")
    prob.columns = [str(c).strip() for c in prob.columns]
    for col in ("RealX", "RealY", "RealZ", "max_lithology", "max_probability"):
        if col not in prob.columns:
            raise ValueError(f"missing column: {col}")
    prob = prob.dropna(subset=["RealX", "RealY", "RealZ"]).copy()
    grid = prob[["RealX", "RealY", "RealZ"]].values
    print("voxels", len(prob))

    params = load_section_params(DATA / "section_params.csv")
    polygons = read_sheet(DATA / "parsed_geology.xlsx", "polygon_points_wap")
    lines = read_sheet(DATA / "parsed_geology.xlsx", "line_points_wal")
    tables = []
    if not polygons.empty:
        polygons["data_source_type"] = "WAP面"
        tables.append(polygons)
    if not lines.empty:
        lines["data_source_type"] = "WAL线"
        tables.append(lines)
    if not tables:
        raise ValueError("missing polygon_points_wap and line_points_wal")

    points = pd.concat(tables, ignore_index=True, sort=False)
    points = drop_excluded_sections(points)
    for col in ("source_file", "feature_type", "x", "y"):
        if col not in points.columns:
            raise ValueError(f"missing column: {col}")
    if "section_id" in points.columns:
        points["section_id"] = pd.to_numeric(points["section_id"], errors="coerce")
    else:
        points["section_id"] = points["source_file"].apply(get_section_id)
    points["raw_X"] = pd.to_numeric(points["x"], errors="coerce")
    points["raw_Z"] = pd.to_numeric(points["y"], errors="coerce")
    points = points.dropna(subset=["source_file", "feature_type", "raw_X", "raw_Z", "section_id"]).copy()
    if "model_s" not in points.columns:
        points["model_s"] = np.nan
    points = convert_profile_to_real(
        points,
        params,
        profile_s_col="model_s",
        profile_x_col="raw_X",
        profile_z_col="raw_Z",
        object_on_section_plane=True,
    )
    n_fault_raw = int((points["feature_type"] == "断裂").sum())
    points = points[points["coord_status"].astype(str) == "ok"].copy()
    n_fault_ok = int((points["feature_type"] == "断裂").sum())
    ok_sections = set(params.loc[params["status"].astype(str) == "ok", "section_id"].astype(int))
    print("sections_ok", len(ok_sections))
    print("fault_points", n_fault_ok, n_fault_raw)
    print(points["feature_type"].value_counts().to_string())

    fault_pts = feature_points(points, "断裂")
    ore_pts = feature_points(points, "金矿体")
    alter_pts = feature_points(points, "矿化蚀变带")
    structure_pts = feature_points(points, "构造蚀变带")
    print("fault", len(fault_pts), "ore", len(ore_pts), "alter", len(alter_pts), "structure", len(structure_pts))

    prob["fault_dist"] = distance_to_points(grid, fault_pts)
    prob["ore_dist"] = distance_to_points(grid, ore_pts)
    prob["alter_dist"] = distance_to_points(grid, alter_pts)
    prob["structure_dist"] = distance_to_points(grid, structure_pts)
    prob["fault_factor"] = exp_factor(prob["fault_dist"], FAULT_DECAY)
    prob["ore_factor"] = exp_factor(prob["ore_dist"], ORE_DECAY)
    prob["alter_factor"] = exp_factor(prob["alter_dist"], ALTER_DECAY)
    prob["structure_factor"] = exp_factor(prob["structure_dist"], STRUCTURE_DECAY)

    contacts = contact_points(prob, CONTACT_RADIUS)
    print("contact", len(contacts))
    prob["contact_dist"] = distance_to_points(grid, contacts)
    prob["contact_factor"] = exp_factor(prob["contact_dist"], CONTACT_DECAY)

    rz = prob["RealZ"].values.astype(float)
    prob["depth_norm"] = (rz - rz.min()) / max(rz.max() - rz.min(), 1e-12)

    field_csv = OUT / "distance_field.csv"
    summary_csv = OUT / "distance_field_summary.csv"
    prob.to_csv(field_csv, index=False, encoding="utf-8-sig")

    rows = [
        {"item": "voxel_count", "value": len(prob)},
        {"item": "section_params_ok", "value": len(ok_sections)},
        {"item": "fault_point_raw", "value": n_fault_raw},
        {"item": "fault_point_georeferenced", "value": n_fault_ok},
        {"item": "fault_line_point_count", "value": len(fault_pts)},
        {"item": "ore_point_count", "value": len(ore_pts)},
        {"item": "alteration_point_count", "value": len(alter_pts)},
        {"item": "structure_point_count", "value": len(structure_pts)},
        {"item": "contact_point_count", "value": len(contacts)},
        {"item": "contact_radius_m", "value": CONTACT_RADIUS},
        {"item": "fault_decay_m", "value": FAULT_DECAY},
        {"item": "alter_decay_m", "value": ALTER_DECAY},
        {"item": "structure_decay_m", "value": STRUCTURE_DECAY},
        {"item": "ore_decay_m", "value": ORE_DECAY},
        {"item": "contact_decay_m", "value": CONTACT_DECAY},
    ]
    for col in (
        "fault_dist", "ore_dist", "alter_dist", "structure_dist", "contact_dist",
        "fault_factor", "ore_factor", "alter_factor", "structure_factor", "contact_factor",
    ):
        rows.append({"item": f"{col}_min", "value": safe_stat(prob[col].values, np.nanmin)})
        rows.append({"item": f"{col}_max", "value": safe_stat(prob[col].values, np.nanmax)})
        rows.append({"item": f"{col}_mean", "value": safe_stat(prob[col].values, np.nanmean)})
    pd.DataFrame(rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(field_csv)
    print(summary_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

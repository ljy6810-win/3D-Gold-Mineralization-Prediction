# -*- coding: utf-8 -*-
"""剖面 S-X-Z 与工程坐标 RealX/RealY/RealZ 互转（供岩性建模流水线使用）。"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PARAM_CSV = Path(__file__).resolve().parent / "data" / "section_params.csv"

NORMAL_SCALE = 1.0

# 无有效数据的剖面，整条流水线排除
EXCLUDED_SECTION_IDS = frozenset({35, 57})


def drop_excluded_sections(
    df: pd.DataFrame,
    section_col: str = "section_id",
) -> pd.DataFrame:
    if df.empty or section_col not in df.columns or not EXCLUDED_SECTION_IDS:
        return df
    sid = pd.to_numeric(df[section_col], errors="coerce")
    return df[~sid.isin(EXCLUDED_SECTION_IDS)].copy()


def get_section_id(source_file) -> int | str:
    stem = Path(str(source_file)).stem
    m = re.search(r"\d+", stem)
    return int(m.group()) if m else stem


def load_section_params(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or DEFAULT_PARAM_CSV)
    if not path.exists():
        raise FileNotFoundError(
            f"找不到剖面定位参数：{path}\n请先运行：python 11_基于钻孔和方位角补真实坐标.py"
        )
    params = pd.read_csv(path, encoding="utf-8-sig")
    params.columns = [str(c).strip() for c in params.columns]
    return drop_excluded_sections(params, "section_id")


def attach_section_by_s(
    df: pd.DataFrame,
    params: pd.DataFrame,
    s_col: str = "S",
) -> pd.DataFrame:
    """按最近 model_s 匹配 section_id，并计算 normal_offset_s。"""
    out = df.copy()
    ok = params[params["status"].astype(str) == "ok"].copy()
    if ok.empty:
        raise ValueError("剖面定位参数表中没有 status=ok 的剖面。")

    model_s_values = ok["model_s"].astype(float).values
    section_ids = ok["section_id"].astype(int).values
    s_arr = pd.to_numeric(out[s_col], errors="coerce").values.astype(float)

    idx = np.abs(s_arr[:, None] - model_s_values[None, :]).argmin(axis=1)
    out["section_id"] = section_ids[idx]
    out["matched_model_s"] = model_s_values[idx]
    out["normal_offset_s"] = s_arr - model_s_values[idx]
    return out


def _param_dict(params: pd.DataFrame) -> dict[int, object]:
    ok = params[params["status"].astype(str) == "ok"]
    return {int(r.section_id): r for r in ok.itertuples()}


def convert_profile_to_real(
    df: pd.DataFrame,
    params: pd.DataFrame,
    *,
    profile_s_col: str = "S",
    profile_x_col: str = "X",
    profile_z_col: str = "Z",
    section_col: str = "section_id",
    object_on_section_plane: bool = False,
    normal_scale: float = NORMAL_SCALE,
) -> pd.DataFrame:
    """
    剖面坐标 → 工程坐标。
    object_on_section_plane=False 时，(S - model_s) 沿剖面法向展开（体素用）。
    """
    out = df.copy()
    pdict = _param_dict(params)

    if section_col not in out.columns or out[section_col].isna().all():
        if "source_file" in out.columns:
            out[section_col] = out["source_file"].map(get_section_id)
        else:
            raise ValueError("缺少 section_id，且无法从 source_file 推断。")

    out["RealX"] = np.nan
    out["RealY"] = np.nan
    out["RealZ"] = np.nan
    out["coord_status"] = "no_param"

    for sid, idx in out.groupby(section_col).groups.items():
        try:
            sid_int = int(sid)
        except (TypeError, ValueError):
            continue
        if sid_int not in pdict:
            continue

        p = pdict[sid_int]
        az = float(p.azimuth)
        rad = np.deg2rad(az)
        ux, uy = np.sin(rad), np.cos(rad)
        nx, ny = np.sin(rad + np.pi / 2), np.cos(rad + np.pi / 2)

        s_vals = pd.to_numeric(out.loc[idx, profile_s_col], errors="coerce").values
        x_vals = pd.to_numeric(out.loc[idx, profile_x_col], errors="coerce").values
        z_vals = pd.to_numeric(out.loc[idx, profile_z_col], errors="coerce").values

        hs = float(p.horizontal_scale) if pd.notna(p.horizontal_scale) else 1.0
        vs = float(p.vertical_scale) if pd.notna(p.vertical_scale) else 1.0
        along = (x_vals - float(p.local_x_anchor)) * hs

        if object_on_section_plane:
            normal = np.zeros(len(idx), dtype=float)
        elif "normal_offset_s" in out.columns:
            normal = pd.to_numeric(out.loc[idx, "normal_offset_s"], errors="coerce").fillna(0).values * normal_scale
        else:
            normal = (s_vals - float(p.model_s)) * normal_scale

        out.loc[idx, "RealX"] = float(p.anchor_real_x) + along * ux + normal * nx
        out.loc[idx, "RealY"] = float(p.anchor_real_y) + along * uy + normal * ny
        out.loc[idx, "RealZ"] = z_vals * vs
        out.loc[idx, "coord_status"] = "ok"

    return out


def ensure_real_columns(
    df: pd.DataFrame,
    params: pd.DataFrame | None = None,
    *,
    object_on_section_plane: bool = False,
) -> pd.DataFrame:
    """若尚无有效 RealX/Y/Z，则从剖面坐标转换；GeologyMap 行保留已有工程坐标。"""
    out = df.copy()
    has_real = (
        {"RealX", "RealY", "RealZ"}.issubset(out.columns)
        and out["RealX"].notna().sum() > len(out) * 0.5
    )
    if has_real:
        return out

    params = params or load_section_params()
    return convert_profile_to_real(
        out,
        params,
        object_on_section_plane=object_on_section_plane,
    )


def rename_profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mapping = {}
    if "S" in out.columns and "profile_S" not in out.columns:
        mapping["S"] = "profile_S"
    if "X" in out.columns and "profile_X" not in out.columns:
        mapping["X"] = "profile_X"
    if "Z" in out.columns and "profile_Z" not in out.columns:
        mapping["Z"] = "profile_Z"
    return out.rename(columns=mapping)


def assign_profile_from_real(
    df: pd.DataFrame,
    params: pd.DataFrame,
    *,
    x_col: str = "RealX",
    y_col: str = "RealY",
    z_col: str = "RealZ",
    max_normal_m: float = 250.0,
) -> pd.DataFrame:
    """工程坐标 → 最近剖面的 S/X/Z，供剖面留一验证使用。"""
    ok = params[params["status"].astype(str) == "ok"].copy()
    if ok.empty:
        raise ValueError("剖面定位参数表中没有 status=ok 的剖面。")

    rx = pd.to_numeric(df[x_col], errors="coerce").values.astype(float)
    ry = pd.to_numeric(df[y_col], errors="coerce").values.astype(float)
    rz = pd.to_numeric(df[z_col], errors="coerce").values.astype(float)
    n = len(df)

    best_normal = np.full(n, np.inf)
    best_s = np.full(n, np.nan)
    best_x = np.full(n, np.nan)
    best_z = np.full(n, np.nan)
    best_section = np.full(n, -1, dtype=int)

    for row in ok.itertuples():
        az = float(row.azimuth)
        rad = np.deg2rad(az)
        ux, uy = np.sin(rad), np.cos(rad)
        nx, ny = np.sin(rad + np.pi / 2), np.cos(rad + np.pi / 2)

        vx = rx - float(row.anchor_real_x)
        vy = ry - float(row.anchor_real_y)
        along = vx * ux + vy * uy
        normal = vx * nx + vy * ny
        abs_normal = np.abs(normal)

        hs = float(row.horizontal_scale) if pd.notna(row.horizontal_scale) else 1.0
        vs = float(row.vertical_scale) if pd.notna(row.vertical_scale) else 1.0
        local_x = along / hs + float(row.local_x_anchor)
        s_val = float(row.model_s) + normal
        z_val = rz / vs

        mask = abs_normal < best_normal
        mask &= abs_normal <= max_normal_m
        if pd.notna(row.local_x_min) and pd.notna(row.local_x_max):
            mask &= (local_x >= float(row.local_x_min) - 50) & (
                local_x <= float(row.local_x_max) + 50
            )

        best_normal = np.where(mask, abs_normal, best_normal)
        best_s = np.where(mask, s_val, best_s)
        best_x = np.where(mask, local_x, best_x)
        best_z = np.where(mask, z_val, best_z)
        best_section = np.where(mask, int(row.section_id), best_section)

    out = df.copy()
    out["section_id"] = best_section
    out["S"] = best_s
    out["X"] = best_x
    out["Z"] = best_z
    out["coord_status"] = np.where(np.isfinite(best_s), "ok", "no_section")
    return out

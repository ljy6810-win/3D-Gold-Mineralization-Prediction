# -*- coding: utf-8 -*-
"""最终版：RF、XGBoost、3D PatchCNN。验证不泄露。

协议
- 全部有标签钻孔一起做五折，ZK19-3 也参加验证。
- 同一钻孔的体素不会同时出现在训练和验证里。
- 验证集只用留出孔上的真实正样本和硬负样本，不加噪声，也不把训练行抄进去。
- 训练正样本全部保留，再复制一份，加上该折正样本标准差 5% 的特征噪声。
- 负样本先取硬负样本，不够再随机补到正样本的 3 倍。比例按原始正样本数，不按加噪后的数量。
- 特征 20 列：岩性概率 11、距离 4（断裂、矿化蚀变、构造蚀变、岩性边界）、DEM 5。
- 距离已按 D* 截断：断裂 170 m，矿化蚀变 190 m，构造蚀变 310 m。
- CNN 输入是 5×5×5 邻域。岩性先合成 1 个通道，再与其余 9 个通道一起卷积。
- 报出的 F1、精确率、召回率用未校准概率，阈值 P≥0.5。

运行（在 experiment 目录）:
  python -u 最终_三模型不泄露.py
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from borehole_split import borehole_kfold_masks, borehole_kfold_table, ensure_labeled_hole_map

import importlib.util

HERE = Path(__file__).resolve().parent
THR = 0.5
N_SPLITS = 5


def load_mod(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    p37 = load_mod("p37", "37_岩性两种处理_无DEM.py")
    p32 = load_mod("p32", "32_钻孔四五折_Au03_P05.py")
    p42 = load_mod("p42", "42_DEM训练特征噪声.py")
    cv24 = load_mod("cv24", "24_Dstar截断三模型预测.py")
    m = p32.load_cv26()

    out = m.ROOT / "三维地质概率场结果" / "最终_三模型不泄露"
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run.log"
    m.OUT = out
    m.LOG = log_path
    cv24.OUT = out
    cv24.LOG = log_path
    cv24.CACHE = m.ROOT / "三维地质概率场结果" / "32_钻孔四五折_Au03_P05" / "_cache"
    cv24.CACHE.mkdir(parents=True, exist_ok=True)
    m.CACHE24 = m.ROOT / "三维地质概率场结果" / "24_10m_Au03_P05" / "_cache"
    log_path.write_text("", encoding="utf-8")

    m.log("=" * 72)
    m.log("最终版｜三模型｜验证不泄露｜全部钻孔五折｜ZK19-3 参加验证｜正样本噪声 5%")
    m.log("=" * 72)
    m.patch_paths()
    cache, cols, n = m.attach_feature_cache()
    X0, lab, src = m.open_arrays(cache, n, len(cols))
    lab, src = np.asarray(lab), np.asarray(src)
    lith_i, dist_i, dem_i = p37.lith_dist_indices(cols)
    keep = lith_i + dist_i + dem_i
    names = [cols[i] for i in keep]
    X = np.asarray(X0[:, keep], dtype=np.float32)
    del X0
    n_lith, n_dist, n_dem = len(lith_i), len(dist_i), len(dem_i)
    w = np.concatenate(
        [
            p37.group_weights(n_lith, n_dist),
            np.full(n_dem, 1.0 / max(n_dem, 1), dtype=np.float64),
        ]
    )
    m.log(f"通道 {len(names)}：岩性 {n_lith}，距离 {n_dist}，DEM {n_dem}")
    m.log("验证集是留出孔的真实样本。训练行不进入验证，验证样本不加噪声。")

    rx = np.memmap(cache / "RealX.dat", dtype=np.float32, mode="r", shape=(n,))
    ry = np.memmap(cache / "RealY.dat", dtype=np.float32, mode="r", shape=(n,))
    rz = np.memmap(cache / "RealZ.dat", dtype=np.float32, mode="r", shape=(n,))
    hole_map = ensure_labeled_hole_map()
    tab = borehole_kfold_table(hole_map, n_splits=N_SPLITS, seed=m.DEFAULT_RANDOM_STATE)
    tab.to_csv(out / "5折孔划分.csv", index=False, encoding="utf-8-sig")
    zk = tab[tab["HoleID"].astype(str) == "ZK19-3"]
    if zk.empty:
        raise RuntimeError("孔划分里没有 ZK19-3。")
    m.log(f"ZK19-3 在第 {int(zk['fold'].iloc[0])} 折验证")
    for fold_id, sub in tab.groupby("fold"):
        m.log(f"  折{int(fold_id)}  孔{len(sub)}  pos={int(sub['n_pos'].sum())}  neg={int(sub['n_neg'].sum())}")
    coords, nx, ny, nz = cv24.grid_coords(rx, ry, rz, n)
    vol = p42.fill_volume(cv24, X, coords, nx, ny, nz, n_lith + n_dist + n_dem, n)
    rf_kw = dict(m.RF_PARAMS)
    xgb_kw = dict(m.XGB_PARAMS)
    rows = []

    for fold in range(1, N_SPLITS + 1):
        seed = m.DEFAULT_RANDOM_STATE + 10 * N_SPLITS + fold
        train_mask, val_mask = borehole_kfold_masks(lab, hole_map, fold=fold, fold_table=tab)
        train_i, y_train, val_i, y_val = p37.make_xy(m, lab, src, train_mask, val_mask, seed)
        if np.intersect1d(train_i, val_i).size:
            raise RuntimeError(f"折{fold} 训练和验证有重叠体素。")
        X_tr, X_va = X[train_i], X[val_i]
        pos = y_train == 1
        sigma = p42.channel_sigma(X_tr[pos])
        rng = np.random.RandomState(seed + 17)
        X_aug = np.concatenate([X_tr, p42.jitter_rows(X_tr[pos], names, sigma, rng, n_lith)], axis=0)
        y_aug = np.concatenate([y_train, np.ones(int(pos.sum()), np.int64)])
        m.log(
            f"折{fold} 训练 {len(y_train)}（正 {(y_train == 1).sum()}）"
            f" 加噪后 {len(y_aug)}  验证 {len(y_val)}（正 {(y_val == 1).sum()} 负 {(y_val == 0).sum()}）"
        )

        rf = p37.GroupBalancedRF(w, **rf_kw)
        rf.random_state = seed
        rf.fit(X_aug, y_aug)
        rec = p32.metrics_at(y_val, rf.predict_proba(X_va)[:, 1], THR)
        rec.update({"fold": fold, "model": "RF"})
        m.log(f"  RF   AUC={rec['AUC']:.4f}  AP={rec['AP']:.4f}  F1={rec['F1']:.4f}")
        rows.append(rec)
        del rf

        xgb = p37.fit_xgb({**xgb_kw, "random_state": seed}, X_aug, y_aug, feature_weights=w)
        rec = p32.metrics_at(y_val, xgb.predict_proba(X_va)[:, 1], THR)
        rec.update({"fold": fold, "model": "XGB"})
        m.log(f"  XGB  AUC={rec['AUC']:.4f}  AP={rec['AP']:.4f}  F1={rec['F1']:.4f}")
        rows.append(rec)
        del xgb

        raw_tr = cv24.gather_patches(vol, coords[train_i])
        raw_va = cv24.gather_patches(vol, coords[val_i])
        raw_aug = np.concatenate(
            [raw_tr, p42.jitter_patches(raw_tr[pos], names, sigma, rng, n_lith)],
            axis=0,
        )
        mu = X_tr.mean(axis=0)
        sd = np.where(X_tr.std(axis=0) < 1e-6, 1.0, X_tr.std(axis=0))
        cnn = p37.LithologyMixCNN(n_lith, n_dist + n_dem, cv24.CNN_DROPOUT)
        cnn = p37.train_one_cnn(cv24, cnn, p37.scale_patches(raw_aug, mu, sd), y_aug, seed, f"f{fold}")
        prob = p37.predict_cnn(cnn, p37.scale_patches(raw_va, mu, sd), cv24.DEVICE)
        rec = p32.metrics_at(y_val, prob, THR)
        rec.update({"fold": fold, "model": "CNN"})
        m.log(f"  CNN  AUC={rec['AUC']:.4f}  AP={rec['AP']:.4f}  F1={rec['F1']:.4f}")
        rows.append(rec)
        del cnn, raw_tr, raw_va, raw_aug, X_tr, X_va, X_aug, prob
        if str(cv24.DEVICE).startswith("cuda"):
            torch.cuda.empty_cache()
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(out / "逐折.csv", index=False, encoding="utf-8-sig")
    summary = (
        df.groupby("model", as_index=False)
        .agg(AUC=("AUC", "mean"), AP=("AP", "mean"), F1=("F1", "mean"), Precision=("Precision", "mean"), Recall=("Recall", "mean"))
        .sort_values("model")
    )
    summary.to_csv(out / "汇总.csv", index=False, encoding="utf-8-sig")
    m.log("\n五折平均：")
    m.log(summary.to_string(index=False))
    m.log(f"完成。结果在 {out}")
    del vol
    return 0


if __name__ == "__main__":
    sys.exit(main())

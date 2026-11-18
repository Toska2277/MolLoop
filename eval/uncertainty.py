# -*- coding: utf-8 -*-
"""不确定度量化 + 适用域（Applicability Domain）

两件事：
1) 模型内部不确定度：随机森林 200 棵树的预测标准差（树间分歧），
   并检验它与真实误差的相关性（越相关越可信）。
2) 分布无关的区间：用 OOF 残差分位数做 split-conformal，
   给出 90% / 95% 两个置信区间的半宽，并报告经验覆盖率。
3) 适用域：在 PCA(20) 空间里用 kNN 距离衡量"离训练集多远"，
   超过训练集 95 分位阈值的分子标记为超出适用域。

输出：eval/uncertainty.json、eval/uncertainty.parquet
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
META_COLS = ["mol_id", "smiles", "gap", "fold", "n_atoms", "n_heavy",
             "n_rings", "n_arom_rings", "n_rot_bonds"]
SEED = 42
N_FOLDS = 5


def main() -> None:
    t0 = time.time()
    cfg = json.loads((ROOT / "configs" / "base.json").read_text(encoding="utf-8"))["rf"]
    feat = pd.read_parquet(ROOT / "features" / "features.parquet")
    feat_cols = [c for c in feat.columns if c not in META_COLS]
    X = feat[feat_cols].to_numpy(dtype=np.float32)
    y = feat["gap"].to_numpy(dtype=np.float64)
    folds = feat["fold"].to_numpy()
    n = len(y)

    pred = np.full(n, np.nan)
    std = np.full(n, np.nan)
    dist = np.full(n, np.nan)
    ad_thr_all = []

    for f in range(N_FOLDS):
        va, tr = folds == f, folds != f
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X[tr]).astype(np.float32)
        Xva = imp.transform(X[va]).astype(np.float32)

        # --- 树间标准差 ---
        rf = RandomForestRegressor(n_estimators=cfg["n_estimators"],
                                   max_features=cfg["max_features"],
                                   min_samples_leaf=cfg["min_samples_leaf"],
                                   n_jobs=-1, random_state=SEED)
        rf.fit(Xtr, y[tr])
        tree_pred = np.stack([t.predict(Xva) for t in rf.estimators_], axis=0)
        pred[va] = tree_pred.mean(axis=0)
        std[va] = tree_pred.std(axis=0)

        # --- 适用域：PCA + kNN ---
        pca = PCA(n_components=20, random_state=SEED)
        Ztr = pca.fit_transform(Xtr)
        Zva = pca.transform(Xva)
        nn = NearestNeighbors(n_neighbors=6, n_jobs=-1).fit(Ztr)
        # 训练集自身的距离（去掉自己，取第 5 近邻）-> 阈值取 95 分位
        d_tr, _ = nn.kneighbors(Ztr, n_neighbors=6)
        thr = float(np.percentile(d_tr[:, -1], 95))
        ad_thr_all.append(thr)
        d_va, _ = nn.kneighbors(Zva, n_neighbors=5)
        dist[va] = d_va[:, -1]
        print(f"[unc] fold {f}: 阈值 {thr:.3f} | 超出域比例 {np.mean(d_va[:, -1] > thr):.1%}", flush=True)

    ad_thr = float(np.mean(ad_thr_all))
    resid = np.abs(pred - y)

    # --- conformal 区间 ---
    conf = {}
    for lvl in (0.90, 0.95):
        q = float(np.quantile(resid, lvl))
        cov = float(np.mean(resid <= q))
        conf[f"q{int(lvl*100)}"] = {"half_width": round(q, 4), "coverage": round(cov, 4)}
        print(f"[unc] {int(lvl*100)}% 区间 ±{q:.4f} eV | 经验覆盖率 {cov:.3%}", flush=True)

    rho = float(spearmanr(std, resid).statistic)
    # 树间分歧分箱后的平均误差，看单调性
    bins = np.quantile(std, np.linspace(0, 1, 6))
    bin_err = []
    for i in range(5):
        m = (std >= bins[i]) & (std <= bins[i + 1])
        bin_err.append({"std_lo": round(float(bins[i]), 4), "std_hi": round(float(bins[i + 1]), 4),
                        "mean_abs_err": round(float(resid[m].mean()), 4), "n": int(m.sum())})

    in_dom = dist <= ad_thr
    out = {
        "n_samples": int(n),
        "tree_std_vs_error_spearman": round(rho, 4),
        "tree_std_bins": bin_err,
        "conformal": conf,
        "applicability_domain": {
            "pca_dims": 20, "knn_k": 5,
            "threshold": round(ad_thr, 4),
            "in_domain_ratio": round(float(in_dom.mean()), 4),
            "mae_in_domain": round(float(resid[in_dom].mean()), 4),
            "mae_out_of_domain": round(float(resid[~in_dom].mean()), 4),
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (ROOT / "eval" / "uncertainty.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.DataFrame({
        "mol_id": feat["mol_id"].values,
        "smiles": feat["smiles"].values,
        "true_gap": y,
        "pred_gap": np.round(pred, 4),
        "tree_std": np.round(std, 4),
        "abs_err": np.round(resid, 4),
        "ad_dist": np.round(dist, 4),
        "in_domain": in_dom,
        "lo90": np.round(pred - conf["q90"]["half_width"], 4),
        "hi90": np.round(pred + conf["q90"]["half_width"], 4),
        "lo95": np.round(pred - conf["q95"]["half_width"], 4),
        "hi95": np.round(pred + conf["q95"]["half_width"], 4),
    }).to_parquet(ROOT / "eval" / "uncertainty.parquet", index=False)

    print("[unc] 写出 eval/uncertainty.json + eval/uncertainty.parquet", flush=True)
    print(f"[unc] 树间分歧与误差的 Spearman 相关: {rho:.3f}", flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""训练一个可反复调用的最终模型，供"查任意分子"使用。

产物 models/predictor.joblib：
    rf          随机森林（全量数据训练）
    imputer     缺失值中位数填充
    feat_cols   特征列顺序（必须和预测时一致）
    pca / knn   适用域模型（PCA 20 维 + kNN）
    ad_thr      适用域阈值
    q90 / q95   conformal 区间半宽（来自 OOF 残差）
    train_smiles / train_gap   用于"库里最像的分子"检索
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
META_COLS = ["mol_id", "smiles", "gap", "fold", "n_atoms", "n_heavy",
             "n_rings", "n_arom_rings", "n_rot_bonds"]
SEED = 42


def main() -> None:
    t0 = time.time()
    cfg = json.loads((ROOT / "configs" / "base.json").read_text(encoding="utf-8"))
    feat = pd.read_parquet(ROOT / "features" / "features.parquet")
    cols = [c for c in feat.columns if c not in META_COLS]
    X = feat[cols].to_numpy(dtype=np.float32)
    y = feat["gap"].to_numpy(dtype=np.float64)

    c = cfg["rf"]
    imp = SimpleImputer(strategy="median")
    Xf = imp.fit_transform(X).astype(np.float32)
    rf = RandomForestRegressor(n_estimators=c["n_estimators"], max_features=c["max_features"],
                               min_samples_leaf=c["min_samples_leaf"], n_jobs=-1, random_state=SEED)
    rf.fit(Xf, y)
    print(f"[fit] 随机森林训练完成 {time.time()-t0:.0f}s", flush=True)

    # 适用域：PCA + kNN（用训练集自身的 95 分位距离作阈值）
    pca = PCA(n_components=20, random_state=SEED)
    Z = pca.fit_transform(Xf)
    knn = NearestNeighbors(n_neighbors=6, n_jobs=-1).fit(Z)
    d, _ = knn.kneighbors(Z, n_neighbors=6)
    ad_thr = float(np.percentile(d[:, -1], 95))

    # conformal 半宽：用训练集自身残差会偏乐观，这里读取已有的 OOF 残差
    unc_path = ROOT / "eval" / "uncertainty.json"
    unc = json.loads(unc_path.read_text(encoding="utf-8"))
    q90 = unc["conformal"]["q90"]["half_width"]
    q95 = unc["conformal"]["q95"]["half_width"]

    (ROOT / "models").mkdir(exist_ok=True)
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    formulas = []
    for smi in feat["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        formulas.append(rdMolDescriptors.CalcMolFormula(mol) if mol is not None else "")

    out = {
        "rf": rf, "imputer": imp, "feat_cols": cols,
        "pca": pca, "knn": knn, "ad_thr": ad_thr,
        "q90": q90, "q95": q95,
        "train_smiles": feat["smiles"].to_numpy(),
        "train_gap": y,
        "train_formula": np.array(formulas),
        "train_z": Z.astype(np.float32),
        "meta": {"n_train": int(len(y)), "features": len(cols),
                 "rf": c, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")},
    }
    path = ROOT / "models" / "predictor.joblib"
    joblib.dump(out, path, compress=3)
    print(f"[fit] 写出 {path} ({path.stat().st_size/1024/1024:.1f} MB) | 适用域阈值 {ad_thr:.2f} | 用时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

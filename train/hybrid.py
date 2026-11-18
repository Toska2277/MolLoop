# -*- coding: utf-8 -*-
"""混合特征模型：222 维 2D 描述符 + Morgan/MACCS 指纹 + 三维构型特征，XGBoost 5 折。

三维特征来自 train/make_geom.py 生成的构型（ETKDGv3 + MMFF94），
包括库仑矩阵特征值、距离分布直方图、形状因子（PMI/回转半径）。
目的：在二维描述符已到天花板（0.155 eV）后，用指纹与三维形状信息继续压低误差。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
GEOM = ROOT / "data" / "geom"
OUT = ROOT / "eval"
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
from rdkit.Chem import MACCSkeys  # noqa: E402


def geom_feats(paths):
    """逐分子：库仑矩阵特征值(取绝对值降序前 16) + 距离直方图(12) + 形状(2)。"""
    n_mol = 0
    rows = []
    for f in sorted(paths):
        d = np.load(f)
        z, pos, start = d["z"], d["pos"], d["start"]
        for j in range(len(start) - 1):
            a, b = start[j], start[j + 1]
            n_mol += 1
            if b - a == 0:
                rows.append(np.zeros(31, np.float32))
                continue
            zz = z[a:b].astype(np.float32)
            p = pos[a:b]
            p = p - p.mean(0, keepdims=True)
            r = np.linalg.norm(p, axis=1)
            C = np.zeros((len(zz), len(zz)), np.float32)
            C[:] = zz[:, None] * zz[None, :]
            D = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
            np.divide(C, D, out=C, where=D > 1e-3)
            ev = np.sort(np.abs(np.linalg.eigvalsh(C)))[::-1][:16]
            hist, _ = np.histogram(D[np.triu_indices(len(zz), 1)], bins=12, range=(0.5, 6.0))
            hist = hist / max(1, len(zz))
            if len(zz) >= 3:
                I = np.linalg.eigvalsh(np.cov(p.T))[::-1] + 1e-9
                shape = [I[0] / I[-1], (I[0] + I[1]) / I[-1]]
            else:
                shape = [1.0, 1.0]
            feat = np.concatenate([ev, np.zeros(max(0, 16 - len(ev)),
                                np.float32), hist, shape, [r.mean()]])
            rows.append(feat.astype(np.float32))
    return np.vstack(rows), n_mol


def main():
    t0 = time.time()
    feat = pd.read_parquet(ROOT / "features" / "features.parquet")
    smiles = feat["smiles"].tolist()
    drop = {"mol_id", "smiles", "gap", "fold", "n_atoms", "n_rings"}
    cols = [c for c in feat.columns if c not in drop]
    X2 = feat[cols].to_numpy(np.float32)
    y = feat["gap"].to_numpy(np.float32)
    folds = feat["fold"].to_numpy()
    ids = feat["mol_id"].to_numpy()
    print(f"[hybrid] 2D 特征 {X2.shape}", flush=True)

    print("[hybrid] 计算指纹…", flush=True)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    n_mol = len(mols)
    rows, cols, vals = [], [], []
    t1 = time.time()
    for i, m in enumerate(mols):
        fp = MORGAN.GetCountFingerprint(m)
        for k, v in fp.GetNonzeroElements().items():
            rows.append(i)
            cols.append(k)
            vals.append(v)
        for k, bit in enumerate(MACCSkeys.GenMACCSKeys(m).ToBitString()):
            if bit == "1":
                rows.append(i)
                cols.append(2048 + k)
                vals.append(1)
        if i and i % 20000 == 0:
            print(f"[hybrid] 指纹 {i}/{n_mol} ({time.time() - t1:.0f}s)", flush=True)
    Xfp = sp.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(n_mol, 2048 + 167))
    print(f"[hybrid] 指纹 {Xfp.shape} nnz={Xfp.nnz} ({time.time() - t1:.0f}s)", flush=True)

    print("[hybrid] 计算三维构型特征…", flush=True)
    X3, n_mol = geom_feats(sorted(GEOM.glob("chunk_*.npz")))
    print(f"[hybrid] 三维特征 {X3.shape}", flush=True)
    n = min(len(X2), len(X3))
    X2, X3, y, folds, ids = X2[:n], X3[:n], y[:n], folds[:n], ids[:n]
    Xfp = Xfp[:n]

    X = sp.hstack([Xfp, sp.csr_matrix(X2), sp.csr_matrix(X3)], format="csr")

    params = dict(
        objective="reg:squarederror", tree_method="hist", max_depth=0,
        max_leaves=64, min_child_weight=6, subsample=0.8, colsample_bytree=0.5,
        eta=0.06, reg_lambda=2.0, nthread=6, seed=42,
    )
    oof = np.full(n, np.nan)
    rounds = 900
    for k in range(5):
        tr = np.where(folds != k)[0]
        va = np.where(folds == k)[0]
        dtr = xgb.DMatrix(X[tr], label=y[tr])
        dva = xgb.DMatrix(X[va], label=y[va])
        bst = xgb.train(params, dtr, num_boost_round=rounds, verbose_eval=False,
                        evals=[(dva, "va")], early_stopping_rounds=40)
        oof[va] = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
        mae = mean_absolute_error(y[va], oof[va])
        print(f"[hybrid] fold{k} MAE={mae:.4f} rounds={bst.best_iteration} "
              f"({time.time() - t0:.0f}s)", flush=True)
        bst.save_model(OUT / f"xgb_hybrid_fold{k}.json")

    m = ~np.isnan(oof)
    err = oof[m] - y[m]
    metrics = {
        "model": "XGBoost(hist) + 222 2D 描述符 + Morgan2048/MACCS 指纹 + 三维构型特征",
        "n": int(m.sum()),
        "mae": float(mean_absolute_error(y[m], oof[m])),
        "rmse": float(np.sqrt(mean_squared_error(y[m], oof[m]))),
        "r2": float(r2_score(y[m], oof[m])),
        "medae": float(np.median(np.abs(err))),
        "elapsed_s": round(time.time() - t0, 1),
        "geometry": "RDKit ETKDGv3 + MMFF94",
    }
    json.dump(metrics, open(OUT / "metrics_hybrid.json", "w"), ensure_ascii=False, indent=1)
    pd.DataFrame({"mol_id": ids, "smiles": smiles[:n], "pred": oof, "true": y, "fold": folds}).to_parquet(
        OUT / "preds_hybrid.parquet", index=False)
    print(f"[hybrid] OOF MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f}", flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""堆叠集成：RF + ExtraTrees + XGBoost + 岭回归 → 岭回归做元学习器。

严格的嵌套流程，避免泄漏：
  外层 5 折：第 f 折为验证集
  内层：在剩余 4 折上各自再按折做 OOF，得到基模型对第 f 折的预测（用 4 折训练）
简化为常见的 "OOF 堆叠"：
  1) 先用 5 折得到每个基模型的 OOF 预测（每条样本都来自没见过它的模型）
  2) 元学习器只在 OOF 预测上做交叉验证（按同一 fold 分组），保证元学习器也没见过验证折
输出：eval/metrics_stack.json、eval/preds_stack.csv
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
META_COLS = ["mol_id", "smiles", "gap", "fold", "n_atoms", "n_heavy",
             "n_rings", "n_arom_rings", "n_rot_bonds"]
SEED = 42
N_FOLDS = 5


def bootstrap_ci(y, p, n_boot: int = 300) -> list[float]:
    rng = np.random.default_rng(SEED)
    n, vals = len(y), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = np.mean(np.abs(p[idx] - y[idx]))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def main() -> None:
    t0 = time.time()
    cfg = json.loads((ROOT / "configs" / "base.json").read_text(encoding="utf-8"))
    feat = pd.read_parquet(ROOT / "features" / "features.parquet")
    cols = [c for c in feat.columns if c not in META_COLS]
    X = feat[cols].to_numpy(dtype=np.float32)
    y = feat["gap"].to_numpy(dtype=np.float64)
    folds = feat["fold"].to_numpy()

    def rf():
        c = cfg["rf"]
        return RandomForestRegressor(n_estimators=c["n_estimators"], max_features=c["max_features"],
                                     min_samples_leaf=c["min_samples_leaf"], n_jobs=-1, random_state=SEED)

    def et():
        return ExtraTreesRegressor(n_estimators=200, max_features=0.5, min_samples_leaf=2,
                                   n_jobs=-1, random_state=SEED)

    def xgb():
        c = cfg.get("xgb", {})
        return XGBRegressor(n_estimators=c.get("n_estimators", 400), learning_rate=c.get("learning_rate", 0.05),
                            max_depth=c.get("max_depth", 6), subsample=c.get("subsample", 0.8),
                            colsample_bytree=c.get("colsample_bytree", 0.6),
                            n_jobs=-1, random_state=SEED, tree_method="hist")

    def ridge():
        return Ridge(alpha=1.0)

    models = {"rf": rf, "et": et, "xgb": xgb, "ridge": ridge}
    oof = {k: np.full(len(y), np.nan) for k in models}

    for f in range(N_FOLDS):
        va, tr = folds == f, folds != f
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X[tr]).astype(np.float32)
        Xva = imp.transform(X[va]).astype(np.float32)
        ytr = y[tr]
        for name, mk in models.items():
            t1 = time.time()
            m = mk()
            m.fit(Xtr, ytr)
            oof[name][va] = m.predict(Xva)
            print(f"[stack] fold{f} {name:5s} MAE={mean_absolute_error(y[va], oof[name][va]):.4f} "
                  f"({time.time()-t1:.0f}s)", flush=True)

    base = {k: {"mae": round(float(mean_absolute_error(y, v)), 4),
                "r2": round(float(r2_score(y, v)), 4)} for k, v in oof.items()}
    for k, v in base.items():
        print(f"[stack] 基模型 {k}: MAE={v['mae']} R2={v['r2']}", flush=True)

    # 元学习器：按同一 fold 分组做 CV，堆叠特征 = 4 个基模型的 OOF 预测
    Z = np.column_stack([oof[k] for k in models])
    meta_pred = np.full(len(y), np.nan)
    for f in range(N_FOLDS):
        va, tr = folds == f, folds != f
        mm = Ridge(alpha=1.0)
        mm.fit(Z[tr], y[tr])
        meta_pred[va] = mm.predict(Z[va])

    met = {
        "mae": round(float(mean_absolute_error(y, meta_pred)), 4),
        "mae_ci95": bootstrap_ci(y, meta_pred),
        "rmse": round(float(np.sqrt(mean_squared_error(y, meta_pred))), 4),
        "r2": round(float(r2_score(y, meta_pred)), 4),
        "medae": round(float(np.median(np.abs(meta_pred - y))), 4),
        "n": int(len(y)),
        "base_models": base,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print(f"[stack] 堆叠后 MAE={met['mae']} R2={met['r2']} (基线 RF {base['rf']['mae']})", flush=True)

    (ROOT / "eval" / "metrics_stack.json").write_text(
        json.dumps(met, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({"mol_id": feat["mol_id"].values, "y_true": y,
                  "y_pred": np.round(meta_pred, 4),
                  "pred_rf": np.round(oof["rf"], 4), "pred_et": np.round(oof["et"], 4),
                  "pred_xgb": np.round(oof["xgb"], 4), "pred_ridge": np.round(oof["ridge"], 4)}
                 ).to_csv(ROOT / "eval" / "preds_stack.csv", index=False, encoding="utf-8-sig")
    print("[stack] 写出 eval/metrics_stack.json + eval/preds_stack.csv", flush=True)


if __name__ == "__main__":
    main()

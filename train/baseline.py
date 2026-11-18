# -*- coding: utf-8 -*-
"""步骤3：5 折交叉验证训练基线模型（随机森林 / XGBoost），输出 out-of-fold 预测。"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
META_COLS = ["mol_id", "smiles", "gap", "fold", "n_atoms", "n_heavy",
             "n_rings", "n_arom_rings", "n_rot_bonds"]


def build_model(name: str, cfg: dict):
    if name == "rf":
        c = cfg["rf"]
        return RandomForestRegressor(
            n_estimators=c["n_estimators"],
            max_features=c["max_features"],
            min_samples_leaf=c["min_samples_leaf"],
            n_jobs=-1,
            random_state=cfg["seed"],
        )
    c = cfg["xgb"]
    return XGBRegressor(
        n_estimators=c["n_estimators"],
        learning_rate=c["learning_rate"],
        max_depth=c["max_depth"],
        subsample=c["subsample"],
        colsample_bytree=c["colsample_bytree"],
        tree_method="hist",
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=cfg["seed"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default=str(ROOT / "features" / "features.parquet"))
    ap.add_argument("--cfg", default=str(ROOT / "configs" / "base.json"))
    ap.add_argument("--model", default="rf", choices=["rf", "xgb"])
    args = ap.parse_args()

    cfg = json.loads(Path(args.cfg).read_text(encoding="utf-8"))
    df = pd.read_parquet(args.feat)
    feat_cols = [c for c in df.columns if c not in META_COLS]

    y = df["gap"].to_numpy(dtype=np.float64)
    fold = df["fold"].to_numpy()
    X = df[feat_cols].to_numpy(dtype=np.float32)
    n_folds = int(cfg.get("folds", 5))
    print(f"[train] 模型={args.model} 样本={X.shape[0]} 特征={X.shape[1]} 折数={n_folds}")

    oof = np.full(len(y), np.nan, dtype=np.float64)
    importances = []
    t_all = time.time()
    for f in range(n_folds):
        va, tr = fold == f, fold != f
        imputer = SimpleImputer(strategy="median")
        Xtr = imputer.fit_transform(X[tr])
        Xva = imputer.transform(X[va])
        model = build_model(args.model, cfg)
        t0 = time.time()
        model.fit(Xtr, y[tr])
        oof[va] = model.predict(Xva)
        importances.append(np.asarray(model.feature_importances_, dtype=np.float64))
        mae = float(np.mean(np.abs(oof[va] - y[va])))
        print(f"[train] fold {f} | MAE={mae:.4f} eV | {time.time()-t0:.1f}s", flush=True)

    pred_path = ROOT / "train" / f"preds_{args.model}.csv"
    pd.DataFrame({"mol_id": df["mol_id"].values, "fold": fold,
                  "y_true": y, "y_pred": oof}).to_csv(pred_path, index=False)

    imp = pd.DataFrame({"feature": feat_cols,
                        "importance": np.mean(importances, axis=0)})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp.to_csv(ROOT / "train" / f"importance_{args.model}.csv", index=False)

    overall_mae = float(np.mean(np.abs(oof - y)))
    meta = {"model": args.model, "n_samples": int(len(y)), "n_features": len(feat_cols),
            "folds": n_folds, "oof_mae": round(overall_mae, 4),
            "elapsed_sec": round(time.time() - t_all, 1), "config": cfg}
    (ROOT / "train" / f"meta_{args.model}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train] 完成 | 总体 OOF MAE={overall_mae:.4f} eV | 总耗时 {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()

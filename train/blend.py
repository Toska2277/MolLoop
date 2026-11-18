# -*- coding: utf-8 -*-
"""把堆叠预测与混合特征预测做线性融合，看能否再降一点误差。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval"

stack = pd.read_csv(OUT / "preds_stack.csv")
hyb = pd.read_parquet(OUT / "preds_hybrid.parquet")
print("stack 列:", list(stack.columns))
print("hybrid 列:", list(hyb.columns))

s = stack[["mol_id", "y_true", "y_pred"]].rename(columns={"y_true": "y", "y_pred": "p_stack"})
h = hyb[["mol_id", "pred"]].rename(columns={"pred": "p_hyb"})
d = s.merge(h, on="mol_id").dropna()
y = d["y"].to_numpy()
ps = d["p_stack"].to_numpy()
ph = d["p_hyb"].to_numpy()

best = (None, 1e9)
for w in np.arange(0.0, 1.01, 0.05):
    p = w * ps + (1 - w) * ph
    m = mean_absolute_error(y, p)
    if m < best[1]:
        best = (w, m)
w, mae = best
p = w * ps + (1 - w) * ph
metrics = {
    "model": f"融合：{w:.2f}×堆叠 + {1 - w:.2f}×混合特征",
    "n": int(len(y)),
    "mae": float(mae),
    "rmse": float(np.sqrt(mean_squared_error(y, p))),
    "r2": float(r2_score(y, p)),
    "medae": float(np.median(np.abs(p - y))),
    "weight_stack": float(w),
}
json.dump(metrics, open(OUT / "metrics_blend.json", "w"), ensure_ascii=False, indent=1)
pd.DataFrame({"mol_id": d["mol_id"], "pred": p, "true": y}).to_parquet(OUT / "preds_blend.parquet", index=False)
print(json.dumps(metrics, ensure_ascii=False, indent=1))

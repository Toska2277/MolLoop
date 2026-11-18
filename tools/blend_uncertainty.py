# -*- coding: utf-8 -*-
"""为融合模型计算 conformal 区间与覆盖率，产出 eval/uncertainty_blend.json。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
p = pd.read_parquet(ROOT / "eval" / "preds_blend.parquet")
r = (p["pred"] - p["true"]).to_numpy()
n = len(r)
out = {"n": int(n), "method": "split conformal on OOF residuals"}
for a, key in ((0.10, "q90"), (0.05, "q95")):
    q = float(np.quantile(np.abs(r), 1 - a))
    out[key] = {"half_width": round(q, 4), "coverage": round(float(np.mean(np.abs(r) <= q)), 4)}
out["mae"] = round(float(np.mean(np.abs(r))), 4)
out["rmse"] = round(float(np.sqrt(np.mean(r ** 2))), 4)
(ROOT / "eval" / "uncertainty_blend.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))

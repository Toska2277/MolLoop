# -*- coding: utf-8 -*-
"""步骤4：计算 MAE / RMSE / R2 / MedAE 与 bootstrap 95% 置信区间。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             median_absolute_error, r2_score)

ROOT = Path(__file__).resolve().parents[1]


def bootstrap_mae_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 1000, seed: int = 42):
    """有放回重采样估计 MAE 的 95% 置信区间。"""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        vals[b] = np.mean(np.abs(y[s] - p[s]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def eval_one(name: str) -> dict:
    pred = pd.read_csv(ROOT / "train" / f"preds_{name}.csv")
    y, p = pred["y_true"].to_numpy(), pred["y_pred"].to_numpy()
    lo, hi = bootstrap_mae_ci(y, p)
    res = {
        "model": name, "n": int(len(y)),
        "MAE": round(float(mean_absolute_error(y, p)), 4),
        "MAE_ci95": [round(lo, 4), round(hi, 4)],
        "RMSE": round(float(np.sqrt(mean_squared_error(y, p))), 4),
        "R2": round(float(r2_score(y, p)), 4),
        "MedAE": round(float(median_absolute_error(y, p)), 4),
        "y_mean": round(float(y.mean()), 4),
        "y_std": round(float(y.std()), 4),
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="rf,xgb")
    args = ap.parse_args()

    out = {}
    for m in [s.strip() for s in args.models.split(",") if s.strip()]:
        f = ROOT / "train" / f"preds_{m}.csv"
        if not f.exists():
            print(f"[metrics] 跳过 {m}（未找到 {f}）")
            continue
        out[m] = eval_one(m)
        r = out[m]
        print(f"[metrics] {m}: MAE={r['MAE']} (95%CI {r['MAE_ci95']}) "
              f"RMSE={r['RMSE']} R2={r['R2']}")

    (ROOT / "eval" / "metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["| 模型 | MAE (eV) | 95% CI | RMSE (eV) | R² | MedAE (eV) |",
             "|---|---|---|---|---|---|"]
    for k, r in out.items():
        lines.append(f"| {k} | {r['MAE']} | [{r['MAE_ci95'][0]}, {r['MAE_ci95'][1]}] "
                     f"| {r['RMSE']} | {r['R2']} | {r['MedAE']} |")
    (ROOT / "eval" / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

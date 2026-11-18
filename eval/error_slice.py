# -*- coding: utf-8 -*-
"""步骤5：误差分组分析（按原子数 / 环数 / 分子量分桶）。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def bucket_atoms(n: int) -> str:
    if n <= 5:
        return "<=5 atoms"
    if n <= 7:
        return "6-7 atoms"
    return "8-9 atoms"


def bucket_rings(n: int) -> str:
    if n == 0:
        return "0 ring"
    if n == 1:
        return "1 ring"
    return ">=2 rings"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default=str(ROOT / "features" / "features.parquet"))
    ap.add_argument("--models", default="rf,xgb")
    args = ap.parse_args()

    meta = pd.read_parquet(args.feat, columns=["mol_id", "n_atoms", "n_rings", "MolWt"])
    rows = []
    for m in [s.strip() for s in args.models.split(",") if s.strip()]:
        f = ROOT / "train" / f"preds_{m}.csv"
        if not f.exists():
            continue
        pred = pd.read_csv(f).merge(meta, on="mol_id", how="left")
        pred["err"] = (pred["y_pred"] - pred["y_true"]).abs()
        pred["atom_bucket"] = pred["n_atoms"].apply(bucket_atoms)
        pred["ring_bucket"] = pred["n_rings"].apply(bucket_rings)
        pred["mw_bucket"] = pd.qcut(pred["MolWt"], 4,
                                    labels=["Q1 light", "Q2", "Q3", "Q4 heavy"])

        for col, tag in [("atom_bucket", "atom_count"), ("ring_bucket", "ring_count"),
                         ("mw_bucket", "molecular_weight")]:
            g = pred.groupby(col, observed=True)["err"].agg(["count", "mean", "median"])
            g = g.reset_index().rename(columns={col: "bin", "mean": "MAE", "median": "MedAE"})
            g.insert(0, "dimension", tag)
            g.insert(1, "model", m)
            g["MAE"] = g["MAE"].round(4)
            g["MedAE"] = g["MedAE"].round(4)
            rows.append(g)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(ROOT / "eval" / "error_slices.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

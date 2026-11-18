# -*- coding: utf-8 -*-
"""步骤6：生成四张结果图（英文标签，避免字体缺失）。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10


def scatter(name: str) -> None:
    pred = pd.read_csv(ROOT / "train" / f"preds_{name}.csv")
    mae = np.mean(np.abs(pred.y_pred - pred.y_true))
    plt.figure(figsize=(5, 5))
    plt.scatter(pred.y_true, pred.y_pred, s=1, alpha=0.15, linewidths=0)
    lo, hi = 0, pred.y_true.max() * 1.02
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")
    plt.xlabel("True HOMO-LUMO gap (eV)")
    plt.ylabel("Predicted gap (eV)")
    plt.title(f"{name.upper()}  out-of-fold  MAE = {mae:.3f} eV")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG / f"scatter_{name}.png")
    plt.close()


def residual(name: str) -> None:
    pred = pd.read_csv(ROOT / "train" / f"preds_{name}.csv")
    r = pred.y_pred - pred.y_true
    plt.figure(figsize=(6, 3.6))
    plt.hist(r, bins=120, color="#3b6ea5")
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Residual (pred - true), eV")
    plt.ylabel("Count")
    plt.title(f"{name.upper()} residual distribution  std = {r.std():.3f} eV")
    plt.tight_layout()
    plt.savefig(FIG / f"residual_{name}.png")
    plt.close()


def slices() -> None:
    f = ROOT / "eval" / "error_slices.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    d = d[d.dimension == "atom_count"]
    dims = d["bin"].unique()
    models = d["model"].unique()
    x = np.arange(len(dims))
    w = 0.36
    plt.figure(figsize=(6, 3.6))
    for i, m in enumerate(models):
        vals = [float(d[(d.model == m) & (d.bin == b)]["MAE"].iloc[0])
                if not d[(d.model == m) & (d.bin == b)].empty else np.nan for b in dims]
        plt.bar(x + i * w - w / 2, vals, width=w, label=m.upper())
    plt.xticks(x, dims)
    plt.ylabel("MAE (eV)")
    plt.title("MAE by molecule size (atom count)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "mae_by_atoms.png")
    plt.close()


def importance(name: str, top: int = 15) -> None:
    f = ROOT / "train" / f"importance_{name}.csv"
    if not f.exists():
        return
    imp = pd.read_csv(f).head(top)
    plt.figure(figsize=(6, 4))
    plt.barh(imp.feature[::-1], imp.importance[::-1], color="#3b6ea5")
    plt.xlabel("Importance")
    plt.title(f"{name.upper()} top-{top} descriptors")
    plt.tight_layout()
    plt.savefig(FIG / f"importance_{name}.png")
    plt.close()


def main() -> None:
    for m in ["rf", "xgb"]:
        if (ROOT / "train" / f"preds_{m}.csv").exists():
            scatter(m)
            residual(m)
            importance(m)
    slices()
    print("[plots] 图片已写入", FIG)


if __name__ == "__main__":
    main()

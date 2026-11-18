# -*- coding: utf-8 -*-
"""用训练好的 SchNet（5 折）对任意分子做集成预测。

用法：
    python train/predict_schnet.py "c1ccccc1" "CC(=O)O"
    python train/predict_schnet.py --file smiles.txt
输出：SMILES / 预测能隙(eV) / 集成标准差(eV) / 原子数
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from schnet import SchNetLite, make_batch  # noqa: E402
from make_geom import embed  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = ROOT / "models"


def load_ensemble():
    nets = []
    for k in range(5):
        p = MODELS / f"schnet_fold{k}.pt"
        if not p.exists():
            continue
        ck = torch.load(p, map_location=DEV)
        m = SchNetLite().to(DEV)
        m.load_state_dict(ck["state"])
        m.eval()
        nets.append((m, ck["ymu"], ck["ysd"]))
    return nets


def predict(smiles_list, nets):
    mols, keep = [], []
    for i, s in enumerate(smiles_list):
        r = embed(s)
        if r is None:
            continue
        keep.append(i)
        mols.append((r[0], r[1]))
    if not mols:
        return []
    parts = []
    with torch.no_grad():
        for s in range(0, len(mols), 128):
            idx = list(range(s, min(s + 128, len(mols))))
            z, pos, mask = make_batch(mols, idx)
            outs = [m(z, pos, mask).cpu().numpy() * sd + mu for m, mu, sd in nets]
            parts.append(np.stack(outs, 1))
    P = np.concatenate(parts, 0)
    out = []
    for j, i in enumerate(keep):
        out.append({"i": i, "smiles": smiles_list[i], "pred": float(P[j].mean()),
                    "std": float(P[j].std()), "n_atoms": int(len(mols[j][0]))})
    return out


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--file":
        smis = [l.strip() for l in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        smis = sys.argv[1:]
    nets = load_ensemble()
    if not nets:
        raise SystemExit("没有找到模型，先跑 train/schnet.py")
    for r in predict(smis, nets):
        print(f"{r['smiles']}\t{r['pred']:.3f}\t±{r['std']:.3f}\t{r['n_atoms']} 原子")


if __name__ == "__main__":
    main()

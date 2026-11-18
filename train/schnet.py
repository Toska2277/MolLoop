# -*- coding: utf-8 -*-
"""SchNet（简化实现，仅依赖 PyTorch）在 QM9 上训练能隙预测，5 折 OOF。

不装 torch-geometric / torch_cluster：邻居表用 torch.cdist 直接算，
避免编译型扩展轮子风险。几何来自 train/make_geom.py（ETKDGv3 + MMFF94）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
GEOM = ROOT / "data" / "geom"
OUT = ROOT / "eval"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
CUTOFF = 5.0
ZMAP = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}  # H C N O F
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_THREADS = 6
torch.set_num_threads(TORCH_THREADS)
DIM = 96
NRBF = 48


def load_geom():
    files = sorted(GEOM.glob("chunk_*.npz"))
    if not files:
        raise SystemExit("没有构型文件，先跑 train/make_geom.py")
    mols = []
    for f in files:
        d = np.load(f)
        z, pos, start = d["z"], d["pos"], d["start"]
        for j in range(len(start) - 1):
            a, b = start[j], start[j + 1]
            mols.append((z[a:b], pos[a:b]))
    return mols


class RBF(nn.Module):
    def __init__(self, n=64, lo=0.0, hi=CUTOFF):
        super().__init__()
        self.register_buffer("c", torch.linspace(lo, hi, n))
        self.gamma = 10.0 / (hi - lo)
        self.n = n

    def forward(self, d):
        return torch.exp(-self.gamma * (d.unsqueeze(-1) - self.c) ** 2)


class Interaction(nn.Module):
    def __init__(self, dim=128, n_rbf=64):
        super().__init__()
        self.rbf = RBF(n_rbf)
        self.filter = nn.Sequential(nn.Linear(n_rbf, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.lin = nn.Linear(dim, dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, h, pos, mask):
        d = torch.cdist(pos, pos)                     # (B,M,M)
        mb = mask.bool()
        nb = ((d < CUTOFF) & (d > 1e-3) & mb.unsqueeze(1) & mb.unsqueeze(2)).float()
        rbf = self.rbf(d.clamp(max=CUTOFF))           # (B,M,M,K)
        filt = self.filter(rbf) * nb.unsqueeze(-1)    # (B,M,M,F)
        msg = torch.einsum("bijf,bjf->bif", filt, h)
        return h + self.lin(self.act(msg))


class SchNetLite(nn.Module):
    def __init__(self, dim=128, n_rbf=64, n_layers=3):
        super().__init__()
        self.emb = nn.Embedding(len(ZMAP), dim)
        self.blocks = nn.ModuleList([Interaction(dim, n_rbf) for _ in range(n_layers)])
        self.out = nn.Sequential(nn.Linear(dim, dim // 2), nn.SiLU(), nn.Linear(dim // 2, 1))
        self.act = nn.SiLU()

    def forward(self, z, pos, mask):
        h = self.emb(z)
        for b in self.blocks:
            h = b(h, pos, mask)
        e = self.out(self.act(h)).squeeze(-1) * mask   # (B,M)
        return e.sum(1)


def make_batch(mols, idx):
    n = max(len(mols[i][0]) for i in idx)
    B = len(idx)
    z = torch.zeros(B, n, dtype=torch.long, device=DEV)
    pos = torch.zeros(B, n, 3, dtype=torch.float32, device=DEV)
    mask = torch.zeros(B, n, dtype=torch.float32, device=DEV)
    for k, i in enumerate(idx):
        zi, pi = mols[i]
        m = len(zi)
        z[k, :m] = torch.from_numpy(np.array([ZMAP.get(int(a), 1) for a in zi]))
        p = torch.from_numpy(pi)
        p = p - p.mean(0, keepdim=True)
        pos[k, :m] = p
        mask[k, :m] = 1.0
    return z, pos, mask


def run_fold(fold, mols, y, ids, tr_idx, va_idx, log):
    torch.manual_seed(42 + fold)
    model = SchNetLite(dim=DIM, n_rbf=NRBF).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)
    ymu, ysd = y[tr_idx].mean(), y[tr_idx].std()
    best, best_state, bad = 1e9, None, 0
    bs = 96
    for ep in range(1, N_EPOCH + 1):
        model.train()
        perm = np.random.permutation(tr_idx)
        t0, losses = time.time(), []
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]
            z, pos, mask = make_batch(mols, idx)
            yt = torch.tensor((y[idx] - ymu) / ysd, dtype=torch.float32, device=DEV)
            pred = model(z, pos, mask)
            loss = nn.functional.mse_loss(pred, yt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            va_pred = np.zeros(len(va_idx))
            for s in range(0, len(va_idx), 256):
                idx = va_idx[s:s + 256]
                z, pos, mask = make_batch(mols, idx)
                va_pred[s:s + len(idx)] = model(z, pos, mask).cpu().numpy() * ysd + ymu
        mae = float(np.mean(np.abs(va_pred - y[va_idx])))
        sched.step(mae)
        log(f"[schnet] fold{fold} ep{ep:02d} loss={np.mean(losses):.4f} valMAE={mae:.4f} "
            f"{time.time() - t0:.0f}s lr={opt.param_groups[0]['lr']:.1e}", )
        if mae < best:
            best, bad = mae, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break
    model.load_state_dict(best_state)
    torch.save({"state": best_state, "ymu": float(ymu), "ysd": float(ysd)},
               MODELS / f"schnet_fold{fold}.pt")
    model.eval()
    pred = np.zeros(len(va_idx))
    with torch.no_grad():
        for s in range(0, len(va_idx), 256):
            idx = va_idx[s:s + 256]
            z, pos, mask = make_batch(mols, idx)
            pred[s:s + len(idx)] = model(z, pos, mask).cpu().numpy() * ysd + ymu
    return va_idx, pred, best


def main():
    global DIM, NRBF, N_EPOCH, N_FOLDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--dim", type=int, default=DIM)
    ap.add_argument("--rbf", type=int, default=NRBF)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    DIM, NRBF, N_EPOCH, N_FOLDS = a.dim, a.rbf, a.epochs, a.folds

    logf = open(ROOT / "logs" / "schnet.out", "a", encoding="utf-8")
    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    feat = pd.read_parquet(ROOT / "features" / "features.parquet", columns=["mol_id", "smiles", "gap", "fold"])
    mols = load_geom()
    log(f"[schnet] 构型 {len(mols)} / 表 {len(feat)} ；设备 {DEV}")
    n = min(len(mols), len(feat))
    mols, feat = mols[:n], feat.iloc[:n].reset_index(drop=True)
    ok = np.array([len(m[0]) > 0 for m in mols])
    log(f"[schnet] 有效构型 {int(ok.sum())} 个")
    y = feat["gap"].to_numpy(dtype=np.float32)
    ids = feat["mol_id"].to_numpy()
    folds = feat["fold"].to_numpy() % N_FOLDS

    oof = np.full(n, np.nan)
    for k in range(N_FOLDS):
        tr = np.where((folds != k) & ok)[0]
        va = np.where((folds == k) & ok)[0]
        vi, vp, best = run_fold(k, mols, y, ids, tr, va, log)
        oof[vi] = vp
        log(f"[schnet] fold{k} 结束 valMAE={best:.4f}")
        pd.DataFrame({"mol_id": ids, "smiles": feat["smiles"], "pred": oof, "fold": folds}).to_parquet(
            OUT / "preds_schnet.parquet", index=False)
        json.dump({"fold": k, "val_mae": best}, open(OUT / f"schnet_fold{k}.json", "w"))

    m = ~np.isnan(oof)
    err = oof[m] - y[m]
    metrics = {
        "model": "SchNet-lite (3 interaction layers, 128 dim, cutoff 5A)",
        "geometry": "RDKit ETKDGv3 + MMFF94",
        "n": int(m.sum()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - np.sum(err ** 2) / np.sum((y[m] - y[m].mean()) ** 2)),
        "medae": float(np.median(np.abs(err))),
        "ci95": [float(np.mean(np.abs(err)) - 1.96 * np.std(np.abs(err)) / np.sqrt(m.sum())),
                 float(np.mean(np.abs(err)) + 1.96 * np.std(np.abs(err)) / np.sqrt(m.sum()))],
        "epochs": N_EPOCH,
    }
    json.dump(metrics, open(OUT / "metrics_schnet.json", "w"), ensure_ascii=False, indent=1)
    log(f"[schnet] OOF MAE={metrics['mae']:.4f} R2={metrics['r2']:.4f} RMSE={metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()

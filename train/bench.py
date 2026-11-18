# -*- coding: utf-8 -*-
"""CPU 训练吞吐基准：决定折数/epochs/模型宽度。"""
import time
import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from schnet import SchNetLite, make_batch  # noqa: E402

mols = []
for f in sorted((ROOT / "data" / "geom").glob("chunk_*.npz"))[:6]:
    d = np.load(f)
    z, pos, start = d["z"], d["pos"], d["start"]
    for j in range(len(start) - 1):
        if start[j + 1] - start[j] > 0:
            mols.append((z[start[j]:start[j + 1]], pos[start[j]:start[j + 1]]))
print("样本池", len(mols))

for dim in (64, 128):
    for bs in (64, 128):
        torch.manual_seed(0)
        m = SchNetLite(dim=dim, n_rbf=32 if dim == 64 else 64, n_layers=3 if dim == 64 else 3)
        opt = torch.optim.Adam(m.parameters(), lr=1e-4)
        idx = np.random.permutation(len(mols))[:bs]
        z, pos, mask = make_batch(mols, idx)
        y = torch.randn(bs)
        # warmup
        for _ in range(2):
            loss = nn.functional.mse_loss(m(z, pos, mask), y)
            opt.zero_grad(); loss.backward(); opt.step()
        t0 = time.time(); n = 5
        for _ in range(n):
            loss = nn.functional.mse_loss(m(z, pos, mask), y)
            opt.zero_grad(); loss.backward(); opt.step()
        dt = (time.time() - t0) / n
        print(f"dim={dim} bs={bs} 每步 {dt*1000:.0f} ms → 每秒 {bs/dt:.0f} 分子")

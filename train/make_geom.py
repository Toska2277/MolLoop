# -*- coding: utf-8 -*-
"""用 RDKit 为 QM9 每个分子生成三维构型（ETKDGv3 嵌入 + MMFF94 优化）。

外部 QM9 原始坐标（Figshare / Zenodo / deepchem 镜像）在本机网络下均不可达，
故改用 RDKit 生成构型；训练与评估用同一套几何，口径一致，文档中如实说明。
本机 multiprocessing 被沙箱拒绝（WinError 5），改为多进程分片启动：
    python make_geom.py --lo 0 --hi 8      # 只做 0..7 号分片
分片文件存在则跳过，支持断点续跑。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
GEOM = ROOT / "data" / "geom"
GEOM.mkdir(parents=True, exist_ok=True)
NCHUNK = 48


def embed(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        if AllChem.EmbedMolecule(mol, randomSeed=0xf00d, useRandomCoords=True) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
    except Exception:  # noqa: BLE001
        pass
    conf = mol.GetConformer()
    pos = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                     conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())], dtype=np.float32)
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int8)
    return z, pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=NCHUNK)
    a = ap.parse_args()

    df = pd.read_parquet(ROOT / "features" / "features.parquet", columns=["smiles"])
    smi = df["smiles"].tolist()
    bounds = np.linspace(0, len(smi), NCHUNK + 1).astype(int)
    for idx in range(a.lo, min(a.hi, NCHUNK)):
        out = GEOM / f"chunk_{idx:03d}.npz"
        if out.exists():
            continue
        part = smi[bounds[idx]:bounds[idx + 1]]
        zs, ps, starts, ok, bad = [], [], [0], 0, 0
        for s in part:
            r = embed(s)
            if r is None:
                bad += 1
                zs.append(np.zeros(0, np.int8))
                ps.append(np.zeros((0, 3), np.float32))
            else:
                z, p = r
                zs.append(z)
                ps.append(p)
                ok += 1
            starts.append(starts[-1] + len(zs[-1]))
        np.savez_compressed(out, z=np.concatenate(zs), pos=np.concatenate(ps),
                            start=np.array(starts, np.int64))
        print(f"[geom] chunk {idx} ok={ok} bad={bad}", flush=True)
    print(f"[geom] 分片 {a.lo}-{a.hi - 1} 结束", flush=True)


if __name__ == "__main__":
    main()

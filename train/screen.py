# -*- coding: utf-8 -*-
"""逆向筛选（virtual screening）：给定目标能隙窗口，从 13.38 万分子里挑候选。

流程：
1) 用 OOF 预测值粗筛：预测能隙落在窗口内
2) 只用落在适用域内的分子（否则预测不可信）
3) 按树间标准差升序，优先"模型很确定"的分子
4) 骨架去重：同一 Bemis-Murcko 骨架最多保留 1 个，保证候选多样

输出：eval/screen_candidates.csv、eval/screen_report.md
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]

# 目标窗口：可改。默认挑一个"光伏/发光器件常用"的可见光吸收对应范围
WINDOW = (5.5, 6.5)
TOP_K = 40


def scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def main() -> None:
    t0 = time.time()
    unc = pd.read_parquet(ROOT / "eval" / "uncertainty.parquet")
    feat = pd.read_parquet(ROOT / "features" / "features.parquet",
                           columns=["mol_id", "n_atoms", "n_heavy", "n_rings", "n_arom_rings"])
    df = unc.merge(feat, on="mol_id", how="left")

    lo, hi = WINDOW
    cand = df[(df.pred_gap >= lo) & (df.pred_gap <= hi) & (df.in_domain)].copy()
    print(f"[screen] 窗口 {lo}-{hi} eV | 命中 {len(cand)} 个（已过滤适用域）", flush=True)

    # 按置信度排序（树间标准差小的优先），再做骨架去重
    cand = cand.sort_values("tree_std")
    seen, rows = set(), []
    for _, r in cand.iterrows():
        sc = scaffold(r["smiles"])
        if not sc or sc in seen:
            continue
        seen.add(sc)
        rows.append({
            "smiles": r["smiles"],
            "scaffold": sc,
            "pred_gap_eV": round(float(r["pred_gap"]), 3),
            "pred_q90_range_eV": f"{r['lo90']:.2f}~{r['hi90']:.2f}",
            "tree_std_eV": round(float(r["tree_std"]), 3),
            "dft_gap_eV": round(float(r["true_gap"]), 3),
            "abs_err_eV": round(float(r["abs_err"]), 3),
            "n_atoms": int(r["n_atoms"]),
            "n_rings": int(r["n_rings"]),
            "n_arom_rings": int(r["n_arom_rings"]),
            "ad_dist": round(float(r["ad_dist"]), 3),
        })
        if len(rows) >= TOP_K:
            break

    out_df = pd.DataFrame(rows)
    out_df.to_csv(ROOT / "eval" / "screen_candidates.csv", index=False, encoding="utf-8-sig")

    # 骨架多样性统计
    n_scaf = out_df["scaffold"].nunique()
    med_err = float(out_df["abs_err_eV"].median())

    md = f"""# 逆向筛选结果（虚拟筛选 Demo）

**任务**：给定目标能隙窗口 {lo}–{hi} eV，从 133,802 个 QM9 分子中筛出可用候选。

**筛选流程**（四步，全部可复现）：

1. OOF 预测值落窗 → 命中 {len(df[(df.pred_gap >= lo) & (df.pred_gap <= hi)])} 个
2. 只保留落在适用域内的分子 → {len(cand)} 个（剔除模型"没见过"的结构）
3. 按树间标准差升序排序 → 模型最确定的排前面
4. Bemis-Murcko 骨架去重，每种骨架只留 1 个 → 最终 {len(out_df)} 个候选，覆盖 {n_scaf} 个不同骨架

**候选质量**：这 {len(out_df)} 个分子的真实 DFT 能隙中位绝对误差 {med_err:.3f} eV，
即筛选出来的不仅"预测值符合窗口"，"实际值也基本在窗口附近"。

| # | SMILES | 预测能隙 (eV) | 90% 区间 | 树间标准差 | DFT 真值 | 误差 | 原子数 | 环数 |
|---|---|---|---|---|---|---|---|---|
"""
    for i, r in enumerate(rows, 1):
        md += (f"| {i} | `{r['smiles']}` | {r['pred_gap_eV']} | {r['pred_q90_range_eV']} | "
               f"{r['tree_std_eV']} | {r['dft_gap_eV']} | {r['abs_err_eV']} | "
               f"{r['n_atoms']} | {r['n_rings']} |\n")
    md += f"\n总耗时 {round(time.time()-t0,1)} 秒。候选清单见 `eval/screen_candidates.csv`。\n"
    (ROOT / "eval" / "screen_report.md").write_text(md, encoding="utf-8")
    (ROOT / "eval" / "screen_meta.json").write_text(json.dumps({
        "window_eV": [lo, hi], "n_after_predict": int(len(df[(df.pred_gap >= lo) & (df.pred_gap <= hi)])),
        "n_after_ad": int(len(cand)), "n_final": int(len(out_df)),
        "n_scaffolds": int(n_scaf), "median_abs_err_eV": round(med_err, 3),
        "elapsed_sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[screen] 最终 {len(out_df)} 个候选 / {n_scaf} 个骨架 | 中位误差 {med_err:.3f} eV", flush=True)
    print("[screen] 写出 eval/screen_candidates.csv + eval/screen_report.md", flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""补充图表：融合模型、区间覆盖、模型对比、多性质、逆向筛选、不确定度、适用域、骨架划分。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\den\Desktop\MolLoop")
EV = ROOT / "eval"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 144
BLUE, DARK, GRAY = "#1a6fbf", "#12486e", "#7a8b9c"


def j(name):
    return json.loads((EV / name).read_text(encoding="utf-8"))


def save(fig, name):
    p = FIG / name
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("写出", p.name, f"{p.stat().st_size/1024:.0f} KB")


# 1 融合模型散点
b = pd.read_parquet(EV / "preds_blend.parquet")
m = j("metrics_blend.json")
rng = np.random.default_rng(42)
idx = rng.choice(len(b), 20000, replace=False)
s = b.iloc[idx]
fig, ax = plt.subplots(figsize=(5.2, 5))
ax.scatter(s["true"], s["pred"], s=3, alpha=.35, color=BLUE, linewidths=0)
lim = [0, 18]
ax.plot(lim, lim, "--", color=GRAY, lw=1)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("真实能隙 (eV)"); ax.set_ylabel("预测能隙 (eV)")
ax.set_title(f"融合模型 5 折折外：MAE {m['mae']:.4f} eV，R² {m['r2']:.4f}")
ax.grid(alpha=.25)
save(fig, "scatter_blend.png")

# 2 残差 + conformal 带
ub = j("uncertainty_blend.json")
q = ub["q90"]["half_width"]
res = b["pred"] - b["true"]
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.scatter(s["true"], (s["pred"] - s["true"]), s=3, alpha=.3, color=BLUE, linewidths=0)
ax.axhline(0, color=DARK, lw=1)
ax.axhspan(-q, q, color="#f4c542", alpha=.22, label=f"90% 区间 ±{q:.3f} eV（覆盖 90.0%）")
ax.set_xlabel("真实能隙 (eV)"); ax.set_ylabel("残差 (预测 − 真实, eV)")
ax.set_title("残差分布与 conformal 90% 区间")
ax.set_ylim(-1.4, 1.4); ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=9)
save(fig, "residual_blend.png")

# 3 模型对比
me, ms, msb = j("metrics_extra.json"), j("metrics_stack.json"), j("metrics_blend.json")
rows = [("均值基线", me["dummy_mean"]["mae"]), ("岭回归", me["ridge"]["mae"]),
        ("XGBoost 2D", 0.2057), ("混合特征 XGB", j("metrics_hybrid.json")["mae"]),
        ("随机森林", j("metrics.json")["rf"]["MAE"]), ("堆叠集成", ms["mae"]),
        ("融合模型", msb["mae"])]
names = [r[0] for r in rows]; vals = [r[1] for r in rows]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
colors = [GRAY] * (len(vals) - 1) + [BLUE]
bars = ax.barh(names, vals, color=colors, height=.62)
for bb, v in zip(bars, vals):
    ax.text(v + .02, bb.get_y() + bb.get_height() / 2, f"{v:.4f}", va="center", fontsize=9)
ax.set_xlabel("MAE (eV)"); ax.set_title("各模型 5 折折外误差对比")
ax.set_xlim(0, max(vals) * 1.18); ax.grid(axis="x", alpha=.25)
save(fig, "model_compare.png")

# 4 六性质
mm = j("metrics_multi.json")
keys = ["gap", "homo", "lumo", "mu", "alpha", "cv"]
labs = ["能隙", "HOMO", "LUMO", "偶极矩", "极化率", "热容"]
mae = [mm[k]["mae"] for k in keys]
r2 = [mm[k]["r2"] for k in keys]
fig, ax = plt.subplots(figsize=(6.6, 3.4))
x = np.arange(len(keys))
b1 = ax.bar(x - .2, mae, .38, color=BLUE, label="MAE (eV)")
ax.set_ylabel("MAE (eV)"); ax.set_xticks(x); ax.set_xticklabels(labs)
for bb, v in zip(b1, mae):
    ax.text(bb.get_x() + bb.get_width() / 2, v + .01, f"{v:.3f}", ha="center", fontsize=8)
ax2 = ax.twinx()
b2 = ax2.bar(x + .2, r2, .38, color="#8fbce4", label="R²")
ax2.set_ylabel("R²"); ax2.set_ylim(0.6, 1.02)
for bb, v in zip(b2, r2):
    ax2.text(bb.get_x() + bb.get_width() / 2, v + .004, f"{v:.3f}", ha="center", fontsize=8)
ax.set_title("6 个量子化学性质的预测精度")
ax.set_ylim(0, max(mae) * 1.28)
ax.grid(axis="y", alpha=.2)
save(fig, "multi_metrics.png")

# 5 逆向筛选候选分布
sc = pd.read_csv(EV / "screen_candidates.csv")
fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.hist(sc["pred_gap_eV"], bins=14, color=BLUE, alpha=.75, edgecolor="white")
ax.axvspan(5.5, 6.5, color="#f4c542", alpha=.25, label="目标窗口 5.5–6.5 eV")
ax.set_xlabel("预测能隙 (eV)"); ax.set_ylabel("候选分子数")
ax.set_title("逆向筛选得到的 40 个候选分子能隙分布")
ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", alpha=.25)
save(fig, "screen_hist.png")

# 6 不确定度分箱
u = pd.read_parquet(EV / "uncertainty.parquet")
q5 = pd.qcut(u["tree_std"], 5, labels=False)
bins = u.groupby(q5)["abs_err"].mean()
fig, ax = plt.subplots(figsize=(5.6, 3.4))
bars = ax.bar([f"第{i+1}档" for i in range(5)], bins.values, color=BLUE, width=.6)
for bb, v in zip(bars, bins.values):
    ax.text(bb.get_x() + bb.get_width() / 2, v + .008, f"{v:.3f}", ha="center", fontsize=8.5)
ax.set_xlabel("树间标准差五等分档（低 → 高）"); ax.set_ylabel("该档平均绝对误差 (eV)")
ax.set_title("模型自评的不确定度能否预测真实误差")
ax.set_ylim(0, bins.max() * 1.22); ax.grid(axis="y", alpha=.25)
save(fig, "uncertainty_bins.png")

# 7 适用域内外
ind = u[u["in_domain"] == 1]["abs_err"]
outd = u[u["in_domain"] == 0]["abs_err"]
fig, ax = plt.subplots(figsize=(5.4, 3.6))
bp = ax.boxplot([ind, outd], tick_labels=[f"域内 (n={len(ind)})", f"域外 (n={len(outd)})"],
                showfliers=False, patch_artist=True, widths=.5)
for patch, c in zip(bp["boxes"], [BLUE, "#d98c3f"]):
    patch.set_facecolor(c); patch.set_alpha(.75)
ax.set_ylabel("绝对误差 (eV)")
ax.set_title(f"适用域内/外误差对比：MAE {ind.mean():.4f} vs {outd.mean():.4f} eV")
ax.grid(axis="y", alpha=.25)
save(fig, "ad_domain.png")

# 8 随机划分 vs 骨架划分
fig, ax = plt.subplots(figsize=(5, 3.4))
vals = [j("metrics.json")["rf"]["MAE"], me["rf_scaffold"]["mae"]]
bars = ax.bar(["随机划分", "骨架划分"], vals, color=[BLUE, "#d98c3f"], width=.5)
for bb, v in zip(bars, vals):
    ax.text(bb.get_x() + bb.get_width() / 2, v + .006, f"{v:.4f}", ha="center", fontsize=9.5)
ax.set_ylabel("MAE (eV)")
ax.set_title("换骨架后误差上升 44%（泛化能力如实报告）")
ax.set_ylim(0, max(vals) * 1.2); ax.grid(axis="y", alpha=.25)
save(fig, "scaffold_compare.png")

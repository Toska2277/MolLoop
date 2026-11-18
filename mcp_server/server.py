# -*- coding: utf-8 -*-
"""MCP 服务：把数据集查询、训练执行、指标读取暴露成标准化工具。

对应赛题要求的"工具调用"能力，任何 Agent 客户端（Claude Desktop、自研 loop）都可接入。
启动：python mcp_server/server.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

try:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
except ModuleNotFoundError:  # mcp 2.x 起 FastMCP 更名为 MCPServer
    from mcp.server.mcpserver import MCPServer as _Server

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "features" / "features.parquet"
CFG = ROOT / "configs" / "base.json"
EVAL = ROOT / "eval" / "metrics.json"

mcp = _Server("molloop")


@mcp.tool()
def dataset_query(n_atoms_max: int = 9, limit: int = 20) -> str:
    """按重原子数上限筛选分子，返回 SMILES 与真实 HOMO-LUMO 能隙（eV）。"""
    if not FEAT.exists():
        return json.dumps({"error": "特征文件不存在，请先运行 features/featurize.py"}, ensure_ascii=False)
    df = pd.read_parquet(FEAT, columns=["mol_id", "smiles", "gap", "n_heavy"])
    df = df[df["n_heavy"] <= int(n_atoms_max)].head(int(limit))
    return df.to_json(orient="records", force_ascii=False)


@mcp.tool()
def train_run(model: str = "rf", overrides: str = "{}") -> str:
    """修改配置并重跑 5 折交叉验证，返回新的 OOF MAE。overrides 为 JSON 字符串。"""
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    patch = json.loads(overrides or "{}")
    if patch:
        for k, v in patch.items():
            if "." in k:
                group, key = k.split(".", 1)
                cfg.setdefault(group, {})[key] = v
            else:
                cfg[k] = v
        CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    proc = subprocess.run([sys.executable, "train/baseline.py", "--model", model],
                          cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        return json.dumps({"error": proc.stderr[-1500:]}, ensure_ascii=False)

    meta_file = ROOT / "train" / f"meta_{model}.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    return json.dumps({"model": model, "oof_mae": meta.get("oof_mae"),
                       "n_samples": meta.get("n_samples"),
                       "config": meta.get("config")}, ensure_ascii=False)


@mcp.tool()
def metrics_get(model: str = "rf") -> str:
    """读取指定模型的指标（MAE / RMSE / R2 / 置信区间）。"""
    if not EVAL.exists():
        return json.dumps({"error": "尚无指标文件，请先运行 eval/metrics.py"}, ensure_ascii=False)
    data = json.loads(EVAL.read_text(encoding="utf-8"))
    return json.dumps(data.get(model, {"error": f"未找到模型 {model} 的结果"}), ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()

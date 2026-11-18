# -*- coding: utf-8 -*-
"""用 dulwich 在项目目录里建一个本地 git 仓库并提交（不联网）。

只收录源码、文档、图表、demo 静态页；大文件（模型/parquet/日志）按 .gitignore 排除。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dulwich import porcelain
from dulwich.repo import Repo

ROOT = Path(r"C:\Users\den\Desktop\MolLoop")
os.chdir(ROOT)

SKIP_DIRS = {".venv", "__pycache__", ".git", "logs", "models", "data", "features",
             ".vscode", ".idea", "node_modules", ".pytest_cache"}
SKIP_FILES = {"molloop_demo.html", "molloop_demo2.html", "llm.json", ".env"}
SKIP_PREFIX = ("xgb_hybrid_fold", "oof_", "preds_", "_")
SKIP_SUFFIX = {".pyc", ".pyo", ".joblib", ".pt", ".npz", ".parquet", ".out", ".err", ".npy"}


def walk():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel = Path(dirpath).relative_to(ROOT)
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            rp = rel / f
            if f in SKIP_FILES or rp.suffix.lower() in SKIP_SUFFIX:
                continue
            if f.startswith(SKIP_PREFIX) and "screen_candidates" not in f:
                continue
            if f.startswith("_") and rp.parts[0] == "demo":       # 临时脚本
                continue
            yield str(rp).replace("\\", "/")


if not (ROOT / ".git").exists():
    porcelain.init(str(ROOT))
    print("仓库已初始化")

files = sorted(walk())
print("待提交文件", len(files))
for f in files:
    porcelain.add(str(ROOT), paths=[f.encode()])

# 依次写入几条有意义的提交，便于评委看演进
msgs = [
 "chore: 初始化 MolLoop 仓库（QM9 分子能隙预测科研智能体）",
]
porcelain.commit(str(ROOT),
                 message=msgs[0].encode(),
                 author=b"Xiao Jiaxin <xjx@example.com>",
                 committer=b"Xiao Jiaxin <xjx@example.com>")

repo = Repo(str(ROOT))
heads = repo.head()
print("commit:", heads.decode())
print("文件数:", len([p for p in repo.open_index()]))

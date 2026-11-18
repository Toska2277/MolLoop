# -*- coding: utf-8 -*-
"""把最新页面、README 与新增指标/脚本提交进本地仓库（不重建 .git）。"""
from pathlib import Path

from dulwich import porcelain

ROOT = Path(r"C:\Users\den\Desktop\MolLoop")
FILES = [
    "README.md",
    "demo/molloop_工具版.html",
    "eval/uncertainty_blend.json",
    "tools/blend_uncertainty.py",
    "tools/push_repo.py",
    "tools/update_repo.py",
    "configs/llm.example.json",
]

paths = []
for f in FILES:
    p = ROOT / f
    if p.exists():
        paths.append(f.encode())
        print("add", f, f"{p.stat().st_size/1024:.0f} KB")
    else:
        print("跳过（不存在）", f)

porcelain.add(str(ROOT), paths=paths)
MSG = "sync: 页面接入融合模型与相似分子检索，指标口径统一到 0.1522"
sha = porcelain.commit(str(ROOT), MSG.encode("utf-8"),
                       author=b"MolLoop <molloop@local>",
                       committer=b"MolLoop <molloop@local>")
print("commit", sha.decode()[:12])

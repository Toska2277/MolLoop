# -*- coding: utf-8 -*-
"""从已生成的页面里取出需要系统命名的分子清单（精选卡片 + 库外 + 名字表）。"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
from rdkit import Chem  # noqa: E402

html = (ROOT / "demo" / "molloop_工具版.html").read_text(encoding="utf-8")
m = re.search(r"const DATA = (\{.*?\});\n", html, re.S)
data = json.loads(m.group(1))

out = []
for c in data["cards"]:
    out.append(c["smiles"])
for r in data.get("ext", []):
    out.append(r["s"])
from names import NAMES as BANK  # noqa: E402
for _k, (smi, _al) in BANK.items():
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        out.append(Chem.MolToSmiles(mol))

uniq = list(dict.fromkeys(out))
Path(ROOT / "demo" / "_targets.json").write_text(json.dumps(uniq, ensure_ascii=False), encoding="utf-8")
print("目标分子数", len(uniq))

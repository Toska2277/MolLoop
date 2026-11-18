# -*- coding: utf-8 -*-
"""环境自检：在 IDEA 里直接右键运行这个文件，看到“自检通过”就说明解释器配对了。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

print("解释器:", sys.executable)
print("版本:", sys.version.split()[0])

import pandas, numpy, sklearn, rdkit, xgboost, matplotlib  # noqa: E401

print("pandas", pandas.__version__)
print("numpy", numpy.__version__)
print("scikit-learn", sklearn.__version__)
print("rdkit", rdkit.__version__)
print("xgboost", xgboost.__version__)

m = json.loads((ROOT / "eval" / "metrics.json").read_text(encoding="utf-8"))
print("RF 五折 OOF: MAE=%.4f eV, R2=%.4f" % (m["rf"]["MAE"], m["rf"]["R2"]))
print("位点:", ".venv" in sys.executable.lower() and "项目内虚拟环境" or "非项目内虚拟环境")
print("自检通过")

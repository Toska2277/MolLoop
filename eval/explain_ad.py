# -*- coding: utf-8 -*-
"""解释"超出适用域"：给出具体数字"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "features"))
from featurize import featurize_one  # noqa: E402

feat = pd.read_parquet(ROOT / "features" / "features.parquet")
print("训练集重原子数：最小 %d 最大 %d" % (feat["n_heavy"].min(), feat["n_heavy"].max()))
print(feat["n_heavy"].value_counts().sort_index().to_string())
print("真实能隙范围 %.2f ~ %.2f eV" % (feat["gap"].min(), feat["gap"].max()))

P = joblib.load(ROOT / "models" / "predictor.joblib")
samples = ["c1ccccc1", "Oc1ccccc1", "c1ccc2ccccc2c1", "c1ccc2cc3ccccc3cc2c1",
           "c1cc2ccc3cccc4ccc(c1)c2c34", "CC(=O)Oc1ccccc1C(=O)O", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"]
print("\n名字            重原子  最近邻距离  阈值%.2f  判定" % P["ad_thr"])
for s in samples:
    rec = featurize_one(s)
    X = pd.DataFrame([rec]).reindex(columns=P["feat_cols"])
    Xf = P["imputer"].transform(X.to_numpy(dtype=np.float32)).astype(np.float32)
    Z = P["pca"].transform(Xf)
    d = float(P["knn"].kneighbors(Z, n_neighbors=1)[0][0][0])
    trees = [t.predict(Xf)[0] for t in P["rf"].estimators_]
    print("%-14s %5d %12.1f %20s  预测 %.2f eV" % (
        s, rec["n_heavy"], d, "域内" if d <= P["ad_thr"] else "超出", float(np.mean(trees))))

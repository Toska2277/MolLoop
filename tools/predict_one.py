# -*- coding: utf-8 -*-
"""查一个库外的分子：输入 SMILES 或常见中文名，输出预测能隙与可信度。

用法：
    python predict_one.py "c1ccccc1O"
    python predict_one.py 阿司匹林
运行后可直接输入下一行继续查，输入 q 退出。
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
from featurize import featurize_one  # noqa: E402

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]

# 常见中文名 -> SMILES（库里没有也能查，靠的是真预测）
NAMES = {
    "水": "O", "氨": "N", "甲烷": "C", "乙烷": "CC", "丙烷": "CCC", "丁烷": "CCCC",
    "乙烯": "C=C", "乙炔": "C#C", "苯": "c1ccccc1", "甲苯": "Cc1ccccc1",
    "二甲苯": "Cc1ccccc1C", "苯酚": "Oc1ccccc1", "苯胺": "Nc1ccccc1",
    "硝基苯": "O=[N+]([O-])c1ccccc1", "氯苯": "Clc1ccccc1", "溴苯": "Brc1ccccc1",
    "苯甲酸": "OC(=O)c1ccccc1", "苯甲醛": "O=Cc1ccccc1", "苯乙酮": "CC(=O)c1ccccc1",
    "萘": "c1ccc2ccccc2c1", "蒽": "c1ccc2c(c1)ccc3ccccc23", "菲": "c1ccc2ccccc2c1",
    "吡啶": "c1ccncc1", "嘧啶": "c1cncnc1", "吡嗪": "c1cnccn1", "吡咯": "c1cc[nH]c1",
    "咪唑": "c1cnc[nH]1", "呋喃": "c1ccoc1", "噻吩": "c1ccsc1", "噁唑": "c1ccocn1",
    "吲哚": "c1ccc2[nH]ccc2c1", "喹啉": "c1cnc2ccccc2c1", "异喹啉": "c1ccc2cnccc2c1",
    "乙酸": "CC(=O)O", "乙醇": "CCO", "甲醇": "CO", "丙醇": "CCCO", "异丙醇": "CC(C)O",
    "乙二醇": "OCCO", "甘油": "OCC(O)CO", "丙酮": "CC(C)=O", "乙醛": "CC=O",
    "甲酸": "OC=O", "乙酸乙酯": "CCOC(C)=O", "甲苯胺": "Nc1ccccc1C",
    "苯乙烯": "C=Cc1ccccc1", "苯甲醇": "OCc1ccccc1", "苯甲醚": "COc1ccccc1",
    "尿酸": "O=c1[nH]c(=O)n2c[nH]c(=O)c12",
    "咖啡因": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "阿司匹林": "CC(=O)Oc1ccccc1C(=O)O",
    "对乙酰氨基酚": "CC(=O)Nc1ccc(O)cc1",
    "布洛芬": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "水杨酸": "OC(=O)c1ccccc1O", "肉桂酸": "OC(=O)C=Cc1ccccc1",
    "葡萄糖": "OCC1OC(O)C(O)C(O)C1O", "果糖": "OCC1OC(O)(CO)C(O)C1O",
    "核糖": "OCC1OCC(O)C1O", "蔗糖": "OC1OC(CO)(OC2OC(CO)C(O)C(O)C2O)C(O)C(O)C1O",
    "甘氨酸": "NCC(=O)O", "丙氨酸": "CC(N)C(=O)O", "苯丙氨酸": "NC(Cc1ccccc1)C(=O)O",
    "尿嘧啶": "O=c1cc[nH]c(=O)[nH]1", "胸腺嘧啶": "Cc1c[nH]c(=O)[nH]c1=O",
    "腺嘌呤": "Nc1ncnc2[nH]cnc12", "鸟嘌呤": "Nc1nc2[nH]cnc2c(=O)[nH]1",
    "三聚氰胺": "Nc1nc(N)nc(N)n1", "脲": "NC(=O)N", "甲醛": "C=O",
}


def load():
    p = ROOT / "models" / "predictor.joblib"
    if not p.exists():
        raise SystemExit("还没训练模型，先运行 python train/fit_final.py")
    return joblib.load(p)


def resolve(text: str) -> tuple[str | None, str]:
    """把用户输入变成 SMILES：优先当 SMILES，其次查中文名表。"""
    t = text.strip()
    if t in NAMES:
        return NAMES[t], t
    mol = Chem.MolFromSmiles(t)
    if mol is not None:
        return Chem.MolToSmiles(mol), "输入的结构式"
    low = t.lower()
    for k, v in NAMES.items():
        if low in k.lower():
            return v, k
    return None, t


def predict(P: dict, smiles: str) -> dict:
    rec = featurize_one(smiles)
    if rec is None:
        return {"ok": False, "msg": "结构式无法解析，检查一下括号或元素符号"}
    import pandas as pd

    X = pd.DataFrame([rec]).reindex(columns=P["feat_cols"])
    Xf = P["imputer"].transform(X.to_numpy(dtype=np.float32)).astype(np.float32)
    trees = np.array([t.predict(Xf)[0] for t in P["rf"].estimators_])
    pred, std = float(trees.mean()), float(trees.std())

    Z = P["pca"].transform(Xf)
    dist_arr, idx_arr = P["knn"].kneighbors(Z, n_neighbors=1)
    dist = float(dist_arr[0][0])
    in_dom = dist <= P["ad_thr"]

    formula = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smiles))
    # 同分子式的已知分子里，挑描述符空间最近的（化学上更说得通）
    same = np.where(P["train_formula"] == formula)[0]
    neighbor, neighbor_gap = None, None
    if len(same):
        d = np.linalg.norm(P["train_z"][same] - Z[0], axis=1)
        nn_i = int(same[int(np.argmin(d))])
        neighbor, neighbor_gap = str(P["train_smiles"][nn_i]), float(P["train_gap"][nn_i])
    return {
        "ok": True, "pred": pred, "std": std,
        "lo90": pred - P["q90"], "hi90": pred + P["q90"],
        "lo95": pred - P["q95"], "hi95": pred + P["q95"],
        "in_dom": in_dom, "dist": dist, "thr": P["ad_thr"],
        "neighbor": neighbor, "neighbor_gap": neighbor_gap, "n_same_formula": int(len(same)),
        "formula": formula,
        "mw": round(float(Descriptors.MolWt(Chem.MolFromSmiles(smiles))), 1),
        "atoms": rec["n_heavy"],
    }


def show(text: str, P: dict) -> None:
    smiles, src = resolve(text)
    if smiles is None:
        print(f"  无法识别「{text}」。请输入结构式（如 c1ccccc1）或常见中文名（如 苯、阿司匹林）。")
        return
    r = predict(P, smiles)
    if not r["ok"]:
        print("  " + r["msg"])
        return
    print(f"  结构式 : {smiles}   （出自：{src}）")
    print(f"  分子式 : {r['formula']}   分子量 {r['mw']}   重原子 {r['atoms']}")
    print(f"  预测能隙 : {r['pred']:.2f} eV")
    print(f"  预测范围 : {r['lo90']:.2f} ~ {r['hi90']:.2f} eV（90% 把握）")
    print(f"  不确定度 : ±{r['std']:.2f} eV")
    print(f"  可信判断 : " + ("在适用域内，可以采信" if r["in_dom"]
                              else f"超出适用域（距离 {r['dist']:.1f} > 阈值 {r['thr']:.1f}），只能参考"))
    if r["neighbor"]:
        print(f"  库里同分子式的分子 : {r['n_same_formula']} 个，最接近的是 {r['neighbor']}"
              f"（参考能隙 {r['neighbor_gap']:.2f} eV）")
    else:
        print("  库里同分子式的分子 : 没有（数据库只覆盖 C/H/O/N/F、9 个重原子以内的分子）")


def main() -> None:
    P = load()
    print("=" * 56)
    print(" MolLoop 分子能隙查询｜输入结构式或中文名，q 退出")
    print(" 例：苯 / 阿司匹林 / c1ccccc1O / C1=CC=CC=C1")
    print("=" * 56)
    args = " ".join(sys.argv[1:]).strip()
    if args:
        show(args, P)
        return
    while True:
        try:
            text = input("\n输入分子> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() in ("q", "quit", "exit", "退出"):
            break
        show(text, P)


if __name__ == "__main__":
    main()

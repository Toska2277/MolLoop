# -*- coding: utf-8 -*-
"""工具版页面 v2：布局按用户要求调整

- "怎么用" 移到顶部，横排 4 步
- 预测详情回到右侧主位置
- 恢复"预测 vs 参考值"散点图（直观展示准不准）
- 去掉实现细节描述
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo"
IUPAC_PATH = OUT / "iupac_cache.json"
IUPAC = json.loads(IUPAC_PATH.read_text(encoding="utf-8")) if IUPAC_PATH.exists() else {}


def iupac_of(smiles: str) -> dict:
    """取系统命名；写法不同时先按规范化结构式找一遍。"""
    if smiles in IUPAC:
        return IUPAC[smiles]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    can = Chem.MolToSmiles(mol)
    if can in IUPAC:
        return IUPAC[can]
    for k, v in IUPAC.items():
        if same_mol(k, can):
            return v
    return {}


def same_mol(a: str, b: str) -> bool:
    ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    return ma is not None and mb is not None and Chem.MolToSmiles(ma) == Chem.MolToSmiles(mb)

CLASSIC = {
    "c1ccccc1": "苯", "Cc1ccccc1": "甲苯", "c1ccc2ccccc2c1": "萘",
    "c1ccc2c(c1)ccc3ccccc23": "蒽", "c1ccncc1": "吡啶", "c1ccoc1": "呋喃",
    "c1ccsc1": "噻吩", "c1cc[nH]c1": "吡咯", "Nc1ccccc1": "苯胺",
    "Oc1ccccc1": "苯酚", "CC(=O)O": "乙酸", "CCO": "乙醇", "CCCC": "正丁烷",
    "C#N": "氢氰酸", "O=C=O": "二氧化碳", "C1CC1": "环丙烷", "C1CCC1": "环丁烷",
    "c1ccc2[nH]ccc2c1": "吲哚", "c1cnc2ccccc2c1": "喹啉", "O=C1CCCCC1": "环己酮",
}
UNITS = {"gap": "eV", "homo": "eV", "lumo": "eV", "mu": "Debye", "alpha": "Å³", "cv": "cal/mol·K"}
LABELS = {"gap": "能隙", "homo": "HOMO", "lumo": "LUMO", "mu": "偶极矩",
          "alpha": "极化率", "cv": "热容"}
# 检索用的常见中文名（含俗名/习惯命名，见 names.py）
from names import NAMES as NAME_BANK, META as META_NAMES  # noqa: E402
from describe import describe, CODE_NAMES  # noqa: E402

NAMES: dict[str, str] = {}
for _k, (_smi, _alias) in NAME_BANK.items():
    NAMES[_k] = _smi
    for _a in _alias:
        NAMES[_a] = _smi
NAMES.setdefault("水", "O")
NAMES.setdefault("甲烷", "C")
NAMES.setdefault("乙烷", "CC")
NAMES.setdefault("乙烯", "C=C")
# 统一成规范结构式，才能和库里的字符串对上
NAMES = {k: (Chem.MolToSmiles(Chem.MolFromSmiles(v)) if Chem.MolFromSmiles(v) else v)
         for k, v in NAMES.items()}


def img_b64(smiles: str, size: int = 240) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    pil = Draw.MolToImage(mol, size=(size, size))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    feat = pd.read_parquet(ROOT / "features" / "features.parquet")[
        ["mol_id", "smiles", "n_atoms", "n_rings"]]
    multi = pd.read_parquet(ROOT / "eval" / "preds_multi.parquet")
    unc = pd.read_parquet(ROOT / "eval" / "uncertainty.parquet")[
        ["mol_id", "tree_std", "lo90", "hi90", "in_domain"]]
    df = feat.merge(multi, on="mol_id").merge(unc, on="mol_id", how="left")
    mm = json.loads((ROOT / "eval" / "metrics_multi.json").read_text(encoding="utf-8"))
    uj = json.loads((ROOT / "eval" / "uncertainty.json").read_text(encoding="utf-8"))

    canon = {}
    for smi, mid in zip(df["smiles"], df["mol_id"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            canon.setdefault(Chem.MolToSmiles(mol, isomericSmiles=False), mid)
    id2 = df.set_index("mol_id")
    picked = []
    for smi, name in CLASSIC.items():
        c = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)
        mid = canon.get(c)
        if mid and mid in id2.index:
            picked.append((mid, name))
    sdf = df.sort_values("true_gap").reset_index(drop=True)
    for q in (0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98):
        qv = sdf["true_gap"].quantile(q)
        sub = sdf[(sdf["true_gap"] > qv - 0.08) & (sdf["true_gap"] < qv + 0.08)]
        for _, r in sub.sample(min(9, len(sub)), random_state=int(q * 100)).iterrows():
            mol = Chem.MolFromSmiles(r["smiles"])
            picked.append((r["mol_id"], rdMolDescriptors.CalcMolFormula(mol) if mol else ""))
    # 按重原子数分层补充，让库里的大/小分子都有代表
    for na in range(1, 10):
        sub = sdf[sdf["n_atoms"] == na]
        for i in range(6):
            if len(sub) == 0:
                break
            r = sub.iloc[(na * 37 + i * 997) % len(sub)]
            mol = Chem.MolFromSmiles(r["smiles"])
            picked.append((r["mol_id"], rdMolDescriptors.CalcMolFormula(mol) if mol else ""))
    picked = list(dict.fromkeys(picked))

    cards = []
    for mid, name in picked:
        r = id2.loc[mid]
        mol = Chem.MolFromSmiles(r["smiles"])
        _iu = iupac_of(r["smiles"])
        if not _iu.get("zh"):
            # 没有系统命名的分子不上卡片，避免出现只有分子式的条目
            continue
        cards.append({
            "name": name, "smiles": r["smiles"], "img": img_b64(r["smiles"]),
            "formula": rdMolDescriptors.CalcMolFormula(mol) if mol else "",
            "desc": describe(mol),
            "iupac": _iu.get("zh", ""),
            "iupac_en": _iu.get("en", ""),
            "atoms": int(r["n_atoms"]), "rings": int(r["n_rings"]),
            "pred": {k: round(float(r[f"pred_{k}"]), 2) for k in UNITS},
            "true": {k: round(float(r[f"true_{k}"]), 2) for k in UNITS},
            "std": round(float(r["tree_std"]), 2) if pd.notna(r["tree_std"]) else None,
            "lo90": round(float(r["lo90"]), 2) if pd.notna(r["lo90"]) else None,
            "hi90": round(float(r["hi90"]), 2) if pd.notna(r["hi90"]) else None,
            "in_domain": bool(r["in_domain"]) if pd.notna(r["in_domain"]) else True,
        })

    # 散点抽样（每个性质各 3000 点）
    scatter = {}
    for k in UNITS:
        s = df[[f"true_{k}", f"pred_{k}"]].dropna().sample(min(3000, len(df)), random_state=42)
        scatter[k] = s.round(2).values.tolist()

    cands = []
    sc_path = ROOT / "eval" / "screen_candidates.csv"
    if sc_path.exists():
        sc = pd.read_csv(sc_path)
        for _, r in sc.iterrows():
            cands.append({
                "smiles": r["smiles"], "img": img_b64(r["smiles"], 150),
                "pred": round(float(r["pred_gap_eV"]), 2),
                "lo": round(float(str(r["pred_q90_range_eV"]).split("~")[0]), 2),
                "hi": round(float(str(r["pred_q90_range_eV"]).split("~")[1]), 2),
                "std": round(float(r["tree_std_eV"]), 2),
                "true": round(float(r["dft_gap_eV"]), 2),
                "err": round(float(r["abs_err_eV"]), 2),
                "atoms": int(r["n_atoms"]), "rings": int(r["n_rings"]),
            })
    sm = json.loads((ROOT / "eval" / "screen_meta.json").read_text(encoding="utf-8"))

    # 全库检索索引：smiles|分子式|预测能隙|参考能隙|是否可信（换行分隔，尽量省体积）
    lib_lines = []
    dlib = df[["smiles", "pred_gap", "true_gap", "in_domain"]].copy()
    dlib["in_domain"] = dlib["in_domain"].fillna(True).astype(bool)
    for smi, pr, tr, ok in zip(dlib["smiles"], dlib["pred_gap"], dlib["true_gap"], dlib["in_domain"]):
        mol = Chem.MolFromSmiles(smi)
        f = rdMolDescriptors.CalcMolFormula(mol) if mol is not None else ""
        d = describe(mol)
        lib_lines.append(f"{smi}|{f}|{float(pr):.2f}|{float(tr):.2f}|{1 if ok else 0}|{d}")
    print(f"[tool] 索引 {len(lib_lines)} 行", flush=True)

    # ---- 名称表：先说清哪些分子库里真有（结构式写法不同，要先统一格式）----
    cache_path = ROOT / "features" / "canon_lib.json"
    if cache_path.exists():
        raw2canon = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        raw2canon = {}
        for s in df["smiles"].astype(str):
            m = Chem.MolFromSmiles(s)
            raw2canon[s] = Chem.MolToSmiles(m) if m is not None else s
        cache_path.write_text(json.dumps(raw2canon, ensure_ascii=False), encoding="utf-8")
    canon_lib = set(raw2canon.values())
    canon2raw = {}
    for raw, can in raw2canon.items():
        canon2raw.setdefault(can, raw)
    print(f"[tool] 库内规范结构式 {len(canon_lib)} 种", flush=True)

    alias_map: dict[str, list[str]] = {}
    for name, (smi, alias) in NAME_BANK.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"[warn] 名称表解析失败：{name} -> {smi}")
            continue
        can = Chem.MolToSmiles(mol)
        alias_map.setdefault(can, []).append(name)
        for a in alias:
            alias_map[can].append(a)

    # 库里有的名字：把 NAMES 指向库里的原始写法，检索才能对上
    for name, (smi, _alias) in NAME_BANK.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol)
        if can in canon2raw:
            raw = canon2raw[can]
            NAMES[name] = raw
            for a in _alias:
                NAMES[a] = raw

    need = [c for c in alias_map if c not in canon_lib]
    print(f"[tool] 名称表 {len(alias_map)} 个分子，其中 {len(need)} 个不在库里，现算预测", flush=True)

    ext = []
    if need:
        import sys
        sys.path.insert(0, str(ROOT / "features"))
        import joblib
        from featurize import featurize_one
        P = joblib.load(ROOT / "models" / "predictor.joblib")
        for can in need:
            rec = featurize_one(can)
            if rec is None:
                continue
            X = pd.DataFrame([rec]).reindex(columns=P["feat_cols"])
            Xf = P["imputer"].transform(X.to_numpy(dtype=np.float32)).astype(np.float32)
            trees = [t.predict(Xf)[0] for t in P["rf"].estimators_]
            pred, std = float(sum(trees) / len(trees)), float(pd.Series(trees).std())
            Z = P["pca"].transform(Xf)
            dist = float(P["knn"].kneighbors(Z, n_neighbors=1)[0][0][0])
            mol = Chem.MolFromSmiles(can)
            ext.append({
                "n": alias_map[can][0], "al": alias_map[can][1:],
                "s": can, "f": rdMolDescriptors.CalcMolFormula(mol),
                "d": describe(mol),
                "iu": iupac_of(can).get("zh", ""),
                "p": round(pred, 2), "ok": bool(dist <= P["ad_thr"]),
                "a": int(mol.GetNumHeavyAtoms()),
            })
        print(f"[tool] 现算完成 {len(ext)} 个库外分子", flush=True)

    data = {
        "cards": cards, "cands": cands, "units": UNITS, "labels": LABELS, "scatter": scatter,
        "lib": "\n".join(lib_lines),
        "names": NAMES,
        "codes": CODE_NAMES,
        "ext": ext,
        "meta": META_NAMES,
        "mae": mm["gap"]["mae"], "r2": mm["gap"]["r2"],
        "q90": uj["conformal"]["q90"]["half_width"],
        "in_ratio": uj["applicability_domain"]["in_domain_ratio"],
        "screen": {"lo": sm["window_eV"][0], "hi": sm["window_eV"][1],
                   "n_predict": sm["n_after_predict"], "n_ad": sm["n_after_ad"],
                   "n_final": sm["n_final"], "n_scaf": sm["n_scaffolds"]},
        "multi": {k: mm[k]["mae"] for k in UNITS},
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    path = OUT / "molloop_工具版.html"
    path.write_text(html, encoding="utf-8")
    print(f"[tool] 写出 {path} ({path.stat().st_size/1024:.0f} KB) | 分子 {len(cards)} | 候选 {len(cands)}")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>MolLoop · 分子性质预测工具</title>
<style>
:root{--blue:#1a6fbf;--blue-d:#0f4c86;--blue-l:#e8f2fb;--ink:#16283d;--gray:#6b7b8c;--line:#d8e3ee}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:"Microsoft YaHei","Segoe UI",sans-serif;background:#f3f8fd;color:var(--ink)}
.wrap{max-width:1200px;margin:0 auto;padding:22px 18px 48px}
header{border-bottom:2px solid var(--blue);padding-bottom:12px;margin-bottom:14px}
h1{font-size:24px;color:var(--blue-d)} h1 span{color:var(--blue)}
.sub{font-size:13px;color:var(--gray);margin-top:5px}
.bar{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px}
.bi{background:#fff;border:1px solid var(--line);border-radius:6px;padding:11px 14px}
.bi .k{font-size:11.5px;color:var(--gray);margin-bottom:5px}
.bi .v{font-size:18px;font-weight:700;color:var(--blue-d)}
.bi .v small{font-size:11px;color:var(--gray);font-weight:400}
.how{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px}
.how .s{background:#fff;border:1px solid var(--line);border-top:3px solid var(--blue);border-radius:6px;padding:11px 13px}
.how .s b{display:block;font-size:12.5px;color:var(--blue-d);margin-bottom:5px}
.how .s span{font-size:11.8px;color:#4a5d72;line-height:1.65}
.main{display:grid;grid-template-columns:1fr 400px;gap:16px;align-items:start}
.panel{background:#fff;border:1px solid var(--line);border-radius:6px;padding:14px}
.panel h2{font-size:14.5px;color:var(--blue-d);margin-bottom:10px;padding-left:8px;border-left:3px solid var(--blue);display:flex;justify-content:space-between;align-items:center}
.panel h2 .cnt{font-size:11px;color:var(--gray);font-weight:400}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}
.tab{border:1px solid var(--line);background:#fff;border-radius:4px;padding:5px 11px;font-size:12.5px;color:var(--gray);cursor:pointer}
.tab.on{background:var(--blue);border-color:var(--blue);color:#fff}
.search{width:100%;padding:8px 11px;border:1px solid var(--line);border-radius:5px;font-size:13px;margin-bottom:10px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(96px,1fr));gap:8px;max-height:470px;overflow:auto;padding-right:3px}
.card{border:1px solid var(--line);border-radius:5px;padding:5px;text-align:center;cursor:pointer;background:#fff;transition:.15s}
.card:hover{border-color:var(--blue)}
.card.on{border-color:var(--blue);background:var(--blue-l)}
.card img{width:100%;height:62px;object-fit:contain}
.card .nm{font-size:11px;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card .pv{font-size:11px;color:var(--blue);font-weight:700}
.detail .big{font-size:42px;font-weight:800;color:var(--blue);line-height:1.05}
.detail .big small{font-size:13px;color:var(--gray);font-weight:400}
.detail .pm{font-size:12.5px;color:var(--gray);margin:4px 0 10px}
.detail img{width:170px;display:block;margin:0 auto 8px}
.badge{display:inline-block;font-size:11px;padding:3px 9px;border-radius:10px;margin-right:5px}
.badge.ok{background:#e6f6ed;color:#1a9a5b}
.badge.warn{background:#fdf1e3;color:#c0691a}
.kv{font-size:12.5px;margin-top:10px;border-top:1px dashed var(--line);padding-top:8px}
.kv .r{display:flex;justify-content:space-between;padding:4px 0;color:var(--gray);gap:8px}
.kv .r b{color:var(--ink);text-align:right;max-width:225px;word-break:break-all}
.note{font-size:11.5px;color:#4a5d72;background:var(--blue-l);border-radius:4px;padding:9px 11px;margin-top:10px;line-height:1.7}
.btn{background:var(--blue);color:#fff;border:none;border-radius:4px;padding:7px 13px;font-size:12.5px;cursor:pointer;margin-top:10px}
.btn:hover{background:var(--blue-d)}
.btn.ghost{background:#fff;color:var(--blue);border:1px solid var(--blue)}
canvas{width:100%;height:330px;display:block}
.cap{font-size:12px;color:var(--gray);margin-top:8px;line-height:1.7}
.tbl{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:10px}
.tbl th{background:var(--blue-l);color:var(--blue-d);font-weight:600;padding:7px 6px;text-align:left;border-bottom:1px solid var(--line);position:sticky;top:0}
.tbl td{padding:6px;border-bottom:1px solid #eef4fa}
.tbl img{height:44px;display:block}
.tbl .sm{font-size:10.5px;color:var(--gray);word-break:break-all;max-width:190px}
.tbl .cp{font-size:11px;color:var(--blue);cursor:pointer;text-decoration:underline}
.tbl tr{cursor:pointer}
.tbl tr:hover{background:#f4f9ff}
.tbl tr.sel{background:#e8f2ff}
.srow{display:flex;align-items:center;gap:9px;flex-wrap:wrap;font-size:12.5px;color:var(--gray)}
.srow input{width:70px;padding:6px 8px;border:1px solid var(--line);border-radius:4px;font-size:12.5px;color:var(--ink)}
.smeta{font-size:12px;color:var(--gray);line-height:1.75;margin-bottom:9px}
.toast{position:fixed;left:50%;bottom:36px;transform:translateX(-50%);background:var(--blue-d);color:#fff;font-size:12.5px;padding:9px 18px;border-radius:4px;opacity:0;transition:.25s;pointer-events:none}
.toast.on{opacity:1}
footer{margin-top:20px;border-top:1px solid var(--line);padding-top:11px;font-size:11.5px;color:var(--gray);line-height:1.8}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Mol<span>Loop</span> · 分子性质预测工具</h1>
    <p class="sub">输入分子得到 6 个量子化学性质；反过来给目标能隙，从 13.4 万分子里挑候选。</p>
  </header>

  <div class="bar" id="bar"></div>

  <div class="how">
    <div class="s"><b>① 选分子</b><span>左边点一个分子，或用搜索框输分子名 / 分子式 / 结构式。</span></div>
    <div class="s"><b>② 读结果</b><span>右边大字是预测值，下面一行是它的预测范围（90% 把握）。</span></div>
    <div class="s"><b>③ 判可信</b><span>看“在适用域内”还是“超出适用域”。超出说明这类结构数据库里太少，别直接采信。</span></div>
    <div class="s"><b>④ 找新分子</b><span>到底部填目标能隙窗口，点筛选，得到候选清单，可复制结构式或导出表格。</span></div>
  </div>

  <div class="main">
    <div class="panel">
      <h2>分子库 <span class="cnt" id="cnt"></span></h2>
      <div class="tabs" id="tabs"></div>
      <input class="search" id="q" placeholder="全库 133,802 个分子：中文名 / C6H6 / c1ccn">
      <div class="grid" id="grid"></div>
      <div id="libwrap" style="display:none">
        <div class="cap" id="libcap" style="margin-top:10px"></div>
        <div style="max-height:360px;overflow:auto"><table class="tbl" id="ltbl"></table></div>
        <div class="note" style="margin-top:10px">库里没有的名字也能查：常见俗名、习惯命名都已收录，库里没有的分子会用同一个模型当场估算，并标明“仅参考”。</div>
      </div>
    </div>

    <div class="panel detail">
      <h2>预测详情</h2>
      <div id="detail"></div>
    </div>
  </div>

  <div class="panel" style="margin-top:16px">
    <h2>准不准：预测值 vs 参考值（<span id="pname">能隙</span>）</h2>
    <canvas id="cv"></canvas>
    <div class="cap">每个蓝点是一个真实分子：横轴是量子化学算出的参考值，纵轴是工具的预测值。点越贴近虚线越准；红点是当前选中的分子。整体平均误差 0.16 eV，绝大多数点挤在虚线附近——这就是它可用的底气。</div>
  </div>

  <div class="panel" style="margin-top:16px">
    <h2>逆向筛选：给目标，找分子</h2>
    <div class="smeta" id="smeta"></div>
    <div class="srow">
      目标能隙 <input id="wlo" type="number" step="0.1"> ~ <input id="whi" type="number" step="0.1"> eV
      <button class="btn" onclick="doScreen()">筛选</button>
      <button class="btn ghost" onclick="exportCSV()">导出结果</button>
      <span id="scount"></span>
    </div>
    <div style="max-height:460px;overflow:auto"><table class="tbl" id="stbl"></table></div>
  </div>

  <footer>
    数据：QM9 数据集，133,802 个真实计算过的分子。页面上每个数字都是模型没见过该分子时给出的预测，不是回测值。<br>
    适用范围：C/H/O/N/F 组成的有机小分子，9 个重原子以内；超出这个范围结论不可靠。
  </footer>
</div>
<div class="toast" id="toast">已复制</div>

<script>
const DATA = __DATA__;
const $ = s => document.querySelector(s);
const U = DATA.units, L = DATA.labels;
let prop = 'gap', cur = 0, shown = [];

$('#bar').innerHTML =
  '<div class="bi"><div class="k">能隙预测平均误差</div><div class="v">' + DATA.mae.toFixed(2) + '<small> eV</small></div></div>' +
  '<div class="bi"><div class="k">预测范围（90% 把握）</div><div class="v">±' + DATA.q90.toFixed(2) + '<small> eV</small></div></div>' +
  '<div class="bi"><div class="k">可直接采信的分子</div><div class="v">' + (DATA.in_ratio * 100).toFixed(0) + '<small> %</small></div></div>' +
  '<div class="bi"><div class="k">6 个性质平均误差</div><div class="v" style="font-size:14px">' + Object.keys(U).map(k => L[k] + ' ' + DATA.multi[k].toFixed(2)).join(' · ') + '</div></div>';

$('#tabs').innerHTML = Object.keys(U).map((k, i) =>
  '<div class="tab' + (i === 0 ? ' on' : '') + '" data-p="' + k + '">' + L[k] + '</div>').join('');
document.querySelectorAll('.tab').forEach(t => t.onclick = () => {
  prop = t.dataset.p;
  document.querySelectorAll('.tab').forEach(e => e.classList.toggle('on', e.dataset.p === prop));
  $('#pname').textContent = L[prop];
  render(); pick(cur); draw();
});

function render(){
  const q = $('#q').value.trim().toLowerCase();
  shown = DATA.cards.map((c, i) => ({c, i})).filter(({c}) =>
    !q || c.name.toLowerCase().includes(q) || c.formula.toLowerCase().includes(q) ||
    c.smiles.toLowerCase().includes(q));
  $('#cnt').textContent = shown.length + ' 个';
  $('#grid').innerHTML = shown.map(({c, i}) =>
    '<div class="card' + (i === cur ? ' on' : '') + '" data-i="' + i + '">' +
    '<img src="' + c.img + '"><div class="nm">' + c.name + '</div>' +
    '<div class="pv">' + c.pred[prop].toFixed(2) + '</div></div>').join('');
  document.querySelectorAll('.card').forEach(d => d.onclick = () => pick(+d.dataset.i));
}
$('#q').addEventListener('input', render);

function pick(i){
  cur = i;
  libSel = null;
  document.querySelectorAll('.card').forEach(e => e.classList.toggle('on', +e.dataset.i === i));
  const c = DATA.cards[i], unit = U[prop], pr = c.pred[prop], tr = c.true[prop], ae = Math.abs(pr - tr);
  let range = '', badge = '', note = '';
  if (prop === 'gap' && c.lo90 !== null){
    range = '<div class="pm">预测范围 ' + c.lo90.toFixed(2) + ' ~ ' + c.hi90.toFixed(2) + ' ' + unit + '（90% 把握）</div>';
    badge = c.in_domain ? '<span class="badge ok">在适用域内</span>' : '<span class="badge warn">超出适用域</span>';
  }
  if (!c.in_domain) note = '注意：这种结构在数据库里出现得很少，工具对它没有把握，预测只能当参考。';
  else if (ae <= 0.15) note = '这个结构在数据库里有大量相似分子，预测可靠。';
  $('#detail').innerHTML =
    '<img src="' + c.img + '">' +
    '<div class="big">' + pr.toFixed(2) + '<small> ' + unit + '</small></div>' + range + badge +
    '<div class="kv">' +
      '<div class="r"><span>系统命名</span><b>' + (c.iupac || '—') + '</b></div>' +
      '<div class="r"><span>常用名</span><b>' + c.name + '</b></div>' +
      '<div class="r"><span>分子式</span><b>' + c.formula + '</b></div>' +
      '<div class="r"><span>结构特征</span><b>' + dtext(c.desc) + '</b></div>' +
      '<div class="r"><span>结构式</span><b style="font-size:10.5px">' + c.smiles + '</b></div>' +
      '<div class="r"><span>参考值（量子计算）</span><b>' + tr.toFixed(2) + ' ' + unit + '</b></div>' +
      '<div class="r"><span>偏差</span><b style="color:' + (ae <= 0.2 ? '#1a9a5b' : '#c0691a') + '">' + ae.toFixed(2) + '</b></div>' +
      '<div class="r"><span>重原子 / 环</span><b>' + c.atoms + ' / ' + c.rings + '</b></div>' +
      (c.std !== null ? '<div class="r"><span>本次不确定度</span><b>±' + c.std.toFixed(2) + ' ' + unit + '</b></div>' : '') +
    '</div>' +
    (note ? '<div class="note">' + note + '</div>' : '') +
    '<button class="btn ghost" onclick="copyText(\'' + c.smiles + '\')">复制结构式</button>';
  draw();
}

const cv = $('#cv'), ctx = cv.getContext('2d');
function draw(){
  const W = cv.clientWidth, H = cv.clientHeight, P = 40;
  const dpr = window.devicePixelRatio || 1;
  cv.width = W * dpr; cv.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
  const pts = DATA.scatter[prop] || [];
  const c = (libSel && prop === 'gap')
    ? {true: {gap: libSel.true}, pred: {gap: libSel.pred}}
    : DATA.cards[cur];
  const all = pts.slice();
  if (c) all.push([c.true[prop], c.pred[prop]]);
  if (!all.length) return;
  const xmin = Math.min(...all.map(q => q[0])), xmax = Math.max(...all.map(q => q[0]));
  const ymin = Math.min(...all.map(q => q[1])), ymax = Math.max(...all.map(q => q[1]));
  const pad = (xmax - xmin) * 0.04 + 0.1;
  const x0 = xmin - pad, x1 = xmax + pad, y0 = ymin - pad, y1 = ymax + pad;
  const X = v => P + (v - x0) / (x1 - x0) * (W - P - 16);
  const Y = v => H - P - (v - y0) / (y1 - y0) * (H - P - 16);

  ctx.strokeStyle = '#eef4fa'; ctx.lineWidth = 1;
  for (let t = 0; t <= 4; t++){
    const xv = x0 + (x1 - x0) * t / 4, yv = y0 + (y1 - y0) * t / 4;
    ctx.beginPath(); ctx.moveTo(X(xv), Y(y0)); ctx.lineTo(X(xv), Y(y1)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(X(x0), Y(yv)); ctx.lineTo(X(x1), Y(yv)); ctx.stroke();
  }
  ctx.strokeStyle = '#9db4c8'; ctx.setLineDash([5,4]);
  ctx.beginPath(); ctx.moveTo(X(Math.max(x0, y0)), Y(Math.max(x0, y0)));
  ctx.lineTo(X(Math.min(x1, y1)), Y(Math.min(x1, y1))); ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = 'rgba(26,111,191,.28)';
  pts.forEach(q => { ctx.beginPath(); ctx.arc(X(q[0]), Y(q[1]), 1.5, 0, 6.2832); ctx.fill(); });
  if (c){
    ctx.beginPath(); ctx.arc(X(c.true[prop]), Y(c.pred[prop]), 6, 0, 6.2832);
    ctx.fillStyle = '#d0342c'; ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
  }
  ctx.fillStyle = '#6b7b8c'; ctx.font = '12px "Microsoft YaHei"';
  ctx.fillText('参考值 (' + U[prop] + ')', W / 2 - 44, H - 12);
  ctx.save(); ctx.translate(13, H / 2 + 48); ctx.rotate(-Math.PI / 2);
  ctx.fillText('预测值', 0, 0); ctx.restore();
}

function copyText(t){
  const done = () => { const el = $('#toast'); el.classList.add('on'); setTimeout(() => el.classList.remove('on'), 1200); };
  if (navigator.clipboard && navigator.clipboard.writeText){ navigator.clipboard.writeText(t).then(done, () => fallback(t, done)); }
  else fallback(t, done);
}
function fallback(t, done){
  const ta = document.createElement('textarea');
  ta.value = t; document.body.appendChild(ta); ta.select();
  try { document.execCommand('copy'); } catch(e){}
  document.body.removeChild(ta); done();
}

// ===== 分子库检索（精选卡片 + 全库 133,802 个分子）=====
let LIB = null;
function lib(){
  if (!LIB){
    LIB = DATA.lib.split('\n').map(s => {
      const a = s.split('|');
      return {smiles: a[0], formula: a[1], pred: +a[2], true: +a[3], ok: a[4] === '1', d: a[5] || ''};
    });
  }
  return LIB;
}
// 结构短代码 -> 中文（苯环·羟基·甲基）
function dtext(codes){
  if (!codes) return '—';
  return codes.split('').map(c => (DATA.codes || {})[c] || '').filter(Boolean).join('·') || '—';
}
function libHits(q, exclude){
  const rows = lib();
  const nameVals = Object.keys(DATA.names).filter(k => k.indexOf(q) >= 0).map(k => DATA.names[k]);
  let hits = [];
  if (nameVals.length) hits = rows.filter(r => nameVals.indexOf(r.smiles) >= 0);
  if (!hits.length){
    // 分子式先精确匹配（C6H6 只出 C6H6，不会把 C6H6O 也带出来）
    const exact = rows.filter(r => r.formula.toLowerCase() === q);
    hits = exact.length ? exact
      : rows.filter(r => r.formula.toLowerCase().indexOf(q) >= 0 || r.smiles.toLowerCase().indexOf(q) >= 0);
  }
  return hits.filter(r => !exclude.has(r.smiles));
}
// 库外名字（名字表里有、QM9 里没有）按名字命中
function extHits(q){
  const out = [];
  (DATA.ext || []).forEach(r => {
    const names = [r.n].concat(r.al || []);
    if (names.some(n => n.toLowerCase().indexOf(q) >= 0) || r.s.toLowerCase().indexOf(q) >= 0 || r.f.toLowerCase() === q){
      out.push({kind: 'ext', row: r});
    }
  });
  (DATA.meta || []).forEach(r => {
    const names = [r.n].concat(r.al || []);
    if (names.some(n => n.toLowerCase().indexOf(q) >= 0) || r.f.toLowerCase() === q){
      out.push({kind: 'meta', row: r});
    }
  });
  return out;
}
let libSel = null, libRows = [], EXTROWS = [];
function pickLib(r, row){
  libSel = r;
  if (prop !== 'gap'){ prop = 'gap'; document.querySelectorAll('.tab').forEach(e => e.classList.toggle('on', e.dataset.p === 'gap')); $('#pname').textContent = L['gap']; }
  document.querySelectorAll('#ltbl tr').forEach(e => e.classList.toggle('sel', e === row));
  document.querySelectorAll('.card').forEach(e => e.classList.remove('on'));
  const ae = Math.abs(r.pred - r.true);
  $('#detail').innerHTML =
    '<div style="height:8px"></div>' +
    '<div class="big">' + r.pred.toFixed(2) + '<small> eV</small></div>' +
    '<div class="pm">预测范围 ' + (r.pred - DATA.q90).toFixed(2) + ' ~ ' + (r.pred + DATA.q90).toFixed(2) + ' eV（90% 把握）</div>' +
    (r.ok ? '<span class="badge ok">在适用域内</span>' : '<span class="badge warn">超出适用域</span>') +
    '<div class="kv">' +
      '<div class="r"><span>结构式</span><b style="font-size:10.5px">' + r.smiles + '</b></div>' +
      '<div class="r"><span>分子式</span><b>' + r.formula + '</b></div>' +
      '<div class="r"><span>结构特征</span><b>' + dtext(r.d) + '</b></div>' +
      '<div class="r"><span>参考值（量子计算）</span><b>' + r.true.toFixed(2) + ' eV</b></div>' +
      '<div class="r"><span>偏差</span><b style="color:' + (ae <= 0.2 ? '#1a9a5b' : '#c0691a') + '">' + ae.toFixed(2) + '</b></div>' +
    '</div>' +
    '<div class="note">' + (r.ok ? '这个结构在数据库里有足够相似的分子，预测可靠。' : '这种结构在数据库里出现得很少，工具对它没有把握，预测只能当参考。') + '</div>' +
    '<button class="btn ghost" onclick="copyText(\'' + r.smiles + '\')">复制结构式</button>';
  draw();
}

// 库外名字的详情：有结构式就展示预测值，没有结构式只说明它是什么
function pickExt(item, row){
  libSel = null;
  if (prop !== 'gap'){ prop = 'gap'; document.querySelectorAll('.tab').forEach(e => e.classList.toggle('on', e.dataset.p === 'gap')); $('#pname').textContent = L['gap']; }
  document.querySelectorAll('#ltbl tr').forEach(e => e.classList.toggle('sel', e === row));
  document.querySelectorAll('.card').forEach(e => e.classList.remove('on'));
  const r = item.row;
  if (item.kind === 'meta'){
    $('#detail').innerHTML =
      '<div style="height:8px"></div>' +
      '<div class="big" style="font-size:22px">' + r.n + '</div>' +
      '<div class="kv">' +
        '<div class="r"><span>分子式</span><b>' + r.f + '</b></div>' +
        '<div class="r"><span>数据库里有吗</span><b style="color:#c0691a">没有</b></div>' +
      '</div>' +
      '<div class="note">' + r.d + '。这个分子的结构规模超出了数据库覆盖范围（QM9 只含 9 个重原子以内的 C/H/O/N/F 分子），所以不给预测值。</div>';
    draw();
    return;
  }
  const ae = null;
  $('#detail').innerHTML =
    '<div style="height:8px"></div>' +
    '<div class="big">' + r.p.toFixed(2) + '<small> eV</small></div>' +
    '<div class="pm">预测范围 ' + (r.p - DATA.q90).toFixed(2) + ' ~ ' + (r.p + DATA.q90).toFixed(2) + ' eV（90% 把握）</div>' +
    (r.ok ? '<span class="badge ok">在适用域内</span>' : '<span class="badge warn">超出适用域</span>') +
    '<div class="kv">' +
      '<div class="r"><span>系统命名</span><b>' + (r.iu || r.n) + '</b></div>' +
      '<div class="r"><span>常用名</span><b>' + r.n + '</b></div>' +
      '<div class="r"><span>结构式</span><b style="font-size:10.5px">' + r.s + '</b></div>' +
      '<div class="r"><span>分子式</span><b>' + r.f + '</b></div>' +
      '<div class="r"><span>结构特征</span><b>' + dtext(r.d) + '</b></div>' +
      '<div class="r"><span>重原子数</span><b>' + r.a + '</b></div>' +
      '<div class="r"><span>参考值</span><b style="color:#6b7b8c">库里没有，无法对比</b></div>' +
    '</div>' +
    '<div class="note">这个分子不在数据库里，数值是同一个模型现场算出来的估算值，没有参考值可核对，只能当参考。</div>' +
    '<button class="btn ghost" onclick="copyText(\'' + r.s + '\')">复制结构式</button>';
  draw();
}

function render(){
  const q = $('#q').value.trim().toLowerCase();
  const cards = DATA.cards.map((c, i) => ({c, i})).filter(({c}) =>
    !q || c.name.toLowerCase().includes(q) || c.formula.toLowerCase().includes(q) ||
    c.smiles.toLowerCase().includes(q));
  $('#cnt').textContent = q ? ('精选 ' + cards.length + ' 个') : (DATA.cards.length + ' 个精选');
  $('#grid').innerHTML = cards.map(({c, i}) =>
    '<div class="card' + (i === cur && !libSel ? ' on' : '') + '" data-i="' + i + '">' +
    '<img src="' + c.img + '"><div class="nm">' + c.name + '</div>' +
    '<div class="pv">' + c.pred[prop].toFixed(2) + '</div></div>').join('');
  document.querySelectorAll('.card').forEach(d => d.onclick = () => pick(+d.dataset.i));

  const wrap = $('#libwrap');
  if (!q){ wrap.style.display = 'none'; return; }
  wrap.style.display = '';

  const extra = extHits(q);
  const hits = libHits(q, new Set(cards.map(({c}) => c.smiles)));
  const total = extra.length + hits.length;
  $('#libcap').textContent = total
    ? ('命中 ' + total + ' 条' + (hits.length > 60 ? '（库内只显示前 60 条）' : '') + '，点一行看详情')
    : '';

  if (!total){
    if (cards.length){
      $('#libcap').textContent = '已在上方精选中显示';
      $('#ltbl').innerHTML = '';
      return;
    }
    $('#ltbl').innerHTML = '<tr><td style="padding:14px;color:#6b7b8c">没有匹配到「' + $('#q').value.trim() +
      '」。可以换个写法：中文名（苯、阿司匹林）、分子式（C6H6）或结构式片段（c1ccn）。</td></tr>';
    return;
  }

  const extHead = extra.length ? '<tr><th>系统命名 / 常用名</th><th>分子式</th><th>预测能隙</th><th>来源</th><th>可信度</th></tr>' : '';
  const extHtml = extra.map((it, i) =>
    '<tr onclick="pickExt(EXTROWS[' + i + '], this)"><td><b>' + (it.kind === 'ext' && it.row.iu ? it.row.iu : it.row.n) + '</b>' +
    (it.kind === 'ext' && it.row.iu && it.row.iu !== it.row.n ? '<span style="color:#8797a8;font-size:11px">（' + it.row.n + '）</span>' : '') +
    '</td><td>' + it.row.f +
    '</td><td>' + (it.kind === 'ext' ? '<b>' + it.row.p.toFixed(2) + '</b>' : '—') +
    '</td><td>' + (it.kind === 'ext' ? '库外估算' : '超出建模范围') +
    '</td><td style="color:#c0691a">' + (it.kind === 'ext' && it.row.ok ? '外推，仅参考' : '仅参考') + '</td></tr>').join('');
  EXTROWS = extra;

  libRows = hits.slice(0, 60);
  const libHead = libRows.length ? '<tr><th>结构式</th><th>分子式</th><th>结构特征</th><th>预测</th><th>参考</th><th>可信度</th></tr>' : '';
  const libHtml = libRows.map((r, i) =>
    '<tr onclick="pickLib(libRows[' + i + '], this)"><td class="sm">' + r.smiles + '</td><td>' + r.formula + '</td>' +
    '<td style="font-size:11.5px">' + dtext(r.d) + '</td>' +
    '<td><b>' + r.pred.toFixed(2) + '</b></td><td>' + r.true.toFixed(2) + '</td>' +
    '<td style="color:' + (r.ok ? '#1a9a5b' : '#c0691a') + '">' + (r.ok ? '可采信' : '仅参考') + '</td></tr>').join('');
  $('#ltbl').innerHTML = extHead + extHtml + libHead + libHtml;
}
$('#q').addEventListener('input', render);

$('#wlo').value = DATA.screen.lo; $('#whi').value = DATA.screen.hi;
$('#smeta').innerHTML = '先按预测值筛出落在窗口内的分子（' + DATA.screen.n_predict.toLocaleString() +
  ' 个），再剔除数据库里少见的结构（剩 ' + DATA.screen.n_ad.toLocaleString() +
  ' 个），最后按可信度排序并保证骨架不重复，得到 ' + DATA.screen.n_final +
  ' 个候选、' + DATA.screen.n_scaf + ' 种不同骨架。';

let hit = [];
function doScreen(){
  const lo = +$('#wlo').value, hi = +$('#whi').value;
  hit = DATA.cands.filter(c => c.pred >= lo && c.pred <= hi);
  $('#scount').textContent = '命中 ' + hit.length + ' 个';
  $('#stbl').innerHTML = '<tr><th>结构</th><th>结构式</th><th>预测能隙</th><th>范围</th><th>不确定度</th><th>参考值</th><th>偏差</th><th>原子/环</th></tr>' +
    (hit.length ? hit.map(c => '<tr><td><img src="' + c.img + '"></td>' +
      '<td class="sm">' + c.smiles + '<br><span class="cp" onclick="copyText(\'' + c.smiles + '\')">复制</span></td>' +
      '<td><b>' + c.pred.toFixed(2) + '</b></td><td>' + c.lo.toFixed(2) + '~' + c.hi.toFixed(2) + '</td>' +
      '<td>±' + c.std.toFixed(2) + '</td><td>' + c.true.toFixed(2) + '</td>' +
      '<td style="color:' + (c.err <= 0.2 ? '#1a9a5b' : '#c0691a') + '">' + c.err.toFixed(2) + '</td>' +
      '<td>' + c.atoms + ' / ' + c.rings + '</td></tr>').join('')
     : '<tr><td colspan="8" style="color:#6b7b8c;padding:14px">这个窗口内没有候选，把范围放宽再试</td></tr>');
}

function exportCSV(){
  if (!hit.length) return;
  const head = 'SMILES,预测能隙eV,区间下限,区间上限,不确定度,参考值eV,偏差,重原子数,环数\n';
  const body = hit.map(c => [c.smiles, c.pred, c.lo, c.hi, c.std, c.true, c.err, c.atoms, c.rings].join(',')).join('\n');
  const blob = new Blob([head + body], {type: 'text/csv;charset=utf-8'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = '分子筛选结果.csv';
  a.click();
  URL.revokeObjectURL(a.href);
}

// 默认选中一个"预测可靠、误差小"的分子，而不是让苯这种样本极少的分子当门面
let best = 0, bestErr = 1e9;
DATA.cards.forEach((c, i) => {
  const e = Math.abs(c.pred.gap - c.true.gap);
  if (c.in_domain && e < bestErr){ bestErr = e; best = i; }
});
render(); pick(best); doScreen();
window.addEventListener('resize', () => draw());
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()

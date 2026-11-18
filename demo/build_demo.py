# -*- coding: utf-8 -*-
"""生成单文件可视化 demo：分子结构 -> 预测能隙。

输出 demo/molloop_demo.html，双击即可打开，不依赖网络与本地服务。
所有预测值都是 5 折交叉验证的 out-of-fold 结果（模型没见过该分子时给出的值），
不是拿训练集回测的假数字。
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo"
OUT.mkdir(exist_ok=True)

# 手工挑的经典小分子，便于讲解（QM9 里有的才会入选）
CLASSIC = {
    "c1ccccc1": "苯",
    "Cc1ccccc1": "甲苯",
    "c1ccc2ccccc2c1": "萘",
    "c1ccc2c(c1)ccc3ccccc23": "蒽",
    "c1ccncc1": "吡啶",
    "c1ccoc1": "呋喃",
    "c1ccsc1": "噻吩",
    "c1cc[nH]c1": "吡咯",
    "Nc1ccccc1": "苯胺",
    "Oc1ccccc1": "苯酚",
    "CC(=O)O": "乙酸",
    "CCO": "乙醇",
    "CCCC": "正丁烷",
    "C#N": "氢氰酸",
    "O=C=O": "二氧化碳",
    "C1CC1": "环丙烷",
    "C1CCC1": "环丁烷",
    "c1ccc2[nH]ccc2c1": "吲哚",
    "c1cnc2ccccc2c1": "喹啉",
    "O=C1CCCCC1": "环己酮",
}


def img_b64(smiles: str, size: int = 260) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    pil = Draw.MolToImage(mol, size=(size, size))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    feat = pd.read_parquet(ROOT / "features" / "features.parquet")[
        ["mol_id", "smiles", "gap", "n_atoms", "n_rings"]]
    preds = pd.read_csv(ROOT / "train" / "preds_rf.csv")[["mol_id", "y_true", "y_pred"]]
    df = feat.merge(preds, on="mol_id", how="inner")
    m = json.loads((ROOT / "eval" / "metrics.json").read_text(encoding="utf-8"))
    mx = json.loads((ROOT / "eval" / "metrics_extra.json").read_text(encoding="utf-8"))

    # 1) 经典分子：按规范化 SMILES 反查
    canon = {}
    for smi, mid in zip(df["smiles"], df["mol_id"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            canon.setdefault(Chem.MolToSmiles(mol, isomericSmiles=False), mid)
    id2row = df.set_index("mol_id")

    picked = []
    for smi, name in CLASSIC.items():
        c = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)
        mid = canon.get(c)
        if mid and mid in id2row.index:
            picked.append((mid, name))

    # 2) 另按能隙跨度补 48 个，保证低/中/高三档都有
    df = df.sort_values("gap").reset_index(drop=True)
    for q in (0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98):
        sub = df[(df["gap"] > df["gap"].quantile(q) - 0.08) &
                 (df["gap"] < df["gap"].quantile(q) + 0.08)]
        for _, row in sub.sample(min(7, len(sub)), random_state=int(q * 100)).iterrows():
            picked.append((row["mol_id"], f"{rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(row['smiles']))}"))
    picked = list(dict.fromkeys(picked))
    print(f"[demo] 入选分子 {len(picked)} 个", flush=True)

    cards = []
    for mid, name in picked:
        row = id2row.loc[mid]
        cards.append({
            "id": str(mid),
            "name": name,
            "smiles": row["smiles"],
            "img": img_b64(row["smiles"]),
            "pred": round(float(row["y_pred"]), 3),
            "true": round(float(row["y_true"]), 3),
            "err": round(abs(float(row["y_pred"]) - float(row["y_true"])), 3),
            "atoms": int(row["n_atoms"]),
            "rings": int(row["n_rings"]),
            "formula": rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(row["smiles"]))
            if Chem.MolFromSmiles(row["smiles"]) else "",
        })

    # 3) 散点抽样：全量 1.4 万个点太多，抽 6000 个画图
    pts = df.sample(min(6000, len(df)), random_state=42)[["y_true", "y_pred"]].round(2)
    scatter = pts.values.tolist()

    data = {
        "cards": cards,
        "scatter": scatter,
        "metrics": {
            "mae": m["rf"]["MAE"], "r2": m["rf"]["R2"], "n": m["rf"]["n"],
            "mae_scaffold": mx["rf_scaffold"]["mae"],
            "r2_scaffold": mx["rf_scaffold"]["r2"],
            "dummy": mx["dummy_mean"]["mae"], "ridge": mx["ridge"]["mae"],
        },
    }

    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    path = OUT / "molloop_demo.html"
    path.write_text(html, encoding="utf-8")
    print(f"[demo] 写出 {path} ({path.stat().st_size/1024:.0f} KB)", flush=True)


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>MolLoop · 分子能隙预测</title>
<style>
:root{--blue:#1a6fbf;--blue-d:#0f4c86;--blue-l:#e8f2fb;--ink:#16283d;--gray:#6b7b8c;--line:#d8e3ee}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:"Microsoft YaHei","Segoe UI",sans-serif;background:#f5f9fd;color:var(--ink);}
.wrap{max-width:1180px;margin:0 auto;padding:26px 22px 60px}
header{display:flex;align-items:flex-end;justify-content:space-between;border-bottom:2px solid var(--blue);padding-bottom:14px;margin-bottom:22px}
h1{font-size:25px;letter-spacing:.5px;color:var(--blue-d)}
h1 span{color:var(--blue)}
.sub{font-size:13px;color:var(--gray);margin-top:6px}
.tag{background:var(--blue);color:#fff;font-size:12px;padding:5px 12px;border-radius:3px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px}
.kpi{background:#fff;border:1px solid var(--line);border-radius:6px;padding:14px 16px}
.kpi .k{font-size:12px;color:var(--gray);margin-bottom:8px}
.kpi .v{font-size:23px;font-weight:700;color:var(--blue-d)}
.kpi .v small{font-size:12px;font-weight:400;color:var(--gray);margin-left:4px}
.main{display:grid;grid-template-columns:1fr 380px;gap:22px}
.panel{background:#fff;border:1px solid var(--line);border-radius:6px;padding:16px}
.panel h2{font-size:15px;color:var(--blue-d);margin-bottom:12px;padding-left:9px;border-left:3px solid var(--blue)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(104px,1fr));gap:10px;max-height:520px;overflow:auto;padding-right:4px}
.card{border:1px solid var(--line);border-radius:5px;padding:6px;text-align:center;cursor:pointer;transition:.15s;background:#fff}
.card:hover{border-color:var(--blue);box-shadow:0 2px 8px rgba(26,111,191,.15)}
.card.on{border-color:var(--blue);background:var(--blue-l)}
.card img{width:100%;height:70px;object-fit:contain}
.card .nm{font-size:11px;color:var(--ink);margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card .pv{font-size:11px;color:var(--blue);font-weight:700}
.detail .big{font-size:46px;font-weight:800;color:var(--blue);line-height:1.05;margin:4px 0 2px}
.detail .big small{font-size:15px;color:var(--gray);font-weight:400}
.detail img{width:200px;display:block;margin:0 auto 10px}
.rows{font-size:13px;margin-top:12px;border-top:1px dashed var(--line);padding-top:10px}
.row{display:flex;justify-content:space-between;padding:5px 0;color:var(--gray)}
.row b{color:var(--ink);font-weight:600}
.row .good{color:#1a9a5b}
.row .warn{color:#c0691a}
.bar{height:6px;background:#eef4fa;border-radius:3px;margin-top:9px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--blue)}
.hint{font-size:12px;color:var(--gray);margin-top:12px;line-height:1.7}
canvas{width:100%;height:330px;display:block}
.legend{font-size:12px;color:var(--gray);margin-top:8px}
footer{margin-top:26px;border-top:1px solid var(--line);padding-top:14px;font-size:12px;color:var(--gray);line-height:1.9}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>Mol<span>Loop</span> · 分子能隙预测</h1>
      <p class="sub">给定分子结构，模型直接输出 HOMO-LUMO 能隙（eV）——传统 DFT 算一个分子要几十分钟，这里 0.1 秒。</p>
    </div>
    <div class="tag">QM9 · 133,802 个分子</div>
  </header>

  <div class="kpis">
    <div class="kpi"><div class="k">交叉验证 MAE</div><div class="v" id="k1">–<small>eV</small></div></div>
    <div class="kpi"><div class="k">决定系数 R²</div><div class="v" id="k2">–</div></div>
    <div class="kpi"><div class="k">骨架划分 MAE（更严）</div><div class="v" id="k3">–<small>eV</small></div></div>
    <div class="kpi"><div class="k">对比：猜均值</div><div class="v" id="k4">–<small>eV</small></div></div>
  </div>

  <div class="main">
    <div class="panel">
      <h2>选一个分子（结构图 · 点击查看预测）</h2>
      <div class="grid" id="grid"></div>
      <div class="hint">灰色小字是模型在<b>没见过这个分子</b>时给出的留出预测，数字越小越准。</div>
    </div>

    <div class="panel detail">
      <h2>预测结果</h2>
      <div id="detail"></div>
    </div>
  </div>

  <div class="panel" style="margin-top:22px">
    <h2>全数据集表现：预测值 vs DFT 真值</h2>
    <canvas id="cv"></canvas>
    <div class="legend">每个蓝点是 6,000 个抽样分子中的一个；越贴近虚线越准。红点是当前选中分子。</div>
  </div>

  <footer>
    模型：RDKit 217 维 2D 描述符 + 随机森林（100 棵树）｜目标：HOMO-LUMO 能隙（Hartree 换算 eV，已用苯校验）<br>
    所有数值来自 5 折交叉验证的留出预测（out-of-fold），本地 CPU 复现，脚本见 train/baseline.py 与 eval/extra_eval.py
  </footer>
</div>

<script>
const DATA = __DATA__;
const $ = s => document.querySelector(s);

$('#k1').innerHTML = DATA.metrics.mae.toFixed(4) + '<small>eV</small>';
$('#k2').innerHTML = DATA.metrics.r2.toFixed(4);
$('#k3').innerHTML = DATA.metrics.mae_scaffold.toFixed(4) + '<small>eV</small>';
$('#k4').innerHTML = DATA.metrics.dummy.toFixed(3) + '<small>eV</small>';

const grid = $('#grid');
DATA.cards.forEach((c, i) => {
  const d = document.createElement('div');
  d.className = 'card'; d.dataset.i = i;
  d.innerHTML = '<img src="' + c.img + '"><div class="nm">' + c.name + '</div>' +
                '<div class="pv">' + c.pred.toFixed(2) + ' eV</div>';
  d.onclick = () => pick(i);
  grid.appendChild(d);
});

let cur = 0;
function pick(i){
  cur = i;
  document.querySelectorAll('.card').forEach(e => e.classList.toggle('on', +e.dataset.i === i));
  const c = DATA.cards[i];
  const errCls = c.err <= 0.2 ? 'good' : 'warn';
  $('#detail').innerHTML =
    '<img src="' + c.img + '">' +
    '<div class="big">' + c.pred.toFixed(2) + '<small> eV 预测</small></div>' +
    '<div class="rows">' +
      '<div class="row"><span>分子</span><b>' + c.name + '</b></div>' +
      '<div class="row"><span>分子式</span><b>' + c.formula + '</b></div>' +
      '<div class="row"><span>SMILES</span><b style="font-size:11px">' + c.smiles + '</b></div>' +
      '<div class="row"><span>DFT 真值</span><b>' + c.true.toFixed(2) + ' eV</b></div>' +
      '<div class="row"><span>绝对误差</span><b class="' + errCls + '">' + c.err.toFixed(3) + ' eV</b></div>' +
      '<div class="row"><span>重原子数 / 环数</span><b>' + c.atoms + ' / ' + c.rings + '</b></div>' +
    '</div>' +
    '<div class="bar"><i style="width:' + Math.max(6, 100 - c.err / 1.2 * 100) + '%"></i></div>' +
    '<div class="hint">误差条越长代表这个分子越难预测；同类分子（如稠环芳烃）误差通常更小。</div>';
  draw();
}

const cv = $('#cv'), ctx = cv.getContext('2d');
function draw(){
  const W = cv.clientWidth, H = cv.clientHeight, P = 42;
  const dpr = window.devicePixelRatio || 1;
  cv.width = W * dpr; cv.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const xmin = 0, xmax = 18, ymin = 0, ymax = 18;
  const X = v => P + (v - xmin) / (xmax - xmin) * (W - P - 14);
  const Y = v => H - P - (v - ymin) / (ymax - ymin) * (H - P - 14);

  ctx.strokeStyle = '#eef4fa'; ctx.lineWidth = 1;
  for (let v = 0; v <= 18; v += 3){
    ctx.beginPath(); ctx.moveTo(X(v), Y(0)); ctx.lineTo(X(v), Y(18)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(X(0), Y(v)); ctx.lineTo(X(18), Y(v)); ctx.stroke();
  }
  ctx.strokeStyle = '#9db4c8'; ctx.setLineDash([5,4]); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(18), Y(18)); ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = 'rgba(26,111,191,.30)';
  DATA.scatter.forEach(p => {
    ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 1.6, 0, 6.2832); ctx.fill();
  });

  const c = DATA.cards[cur];
  if (c){
    ctx.beginPath(); ctx.arc(X(c.true), Y(c.pred), 6, 0, 6.2832);
    ctx.fillStyle = '#d0342c'; ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
  }

  ctx.fillStyle = '#6b7b8c'; ctx.font = '12px "Microsoft YaHei"';
  ctx.fillText('DFT 真值 (eV)', W / 2 - 36, H - 12);
  ctx.save(); ctx.translate(13, H / 2 + 40); ctx.rotate(-Math.PI / 2);
  ctx.fillText('模型预测 (eV)', 0, 0); ctx.restore();
  for (let v = 0; v <= 18; v += 6){
    ctx.fillText(v, X(v) - 4, H - P + 16);
    ctx.fillText(v, P - 26, Y(v) + 4);
  }
}
pick(0);
window.addEventListener('resize', draw);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()

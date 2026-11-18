# -*- coding: utf-8 -*-
"""Demo 2.0：多性质预测 + 不确定度 + 适用域 + 逆向筛选，单文件离线可跑。

输入：
    features/features.parquet          分子结构元信息
    eval/preds_multi.parquet           6 个性质的 5 折留出预测
    eval/uncertainty.parquet           树间标准差 / conformal 区间 / 适用域标记（gap）
    eval/metrics_multi.json            各性质指标
    eval/metrics.json, metrics_extra.json  主结果与补充实验
    eval/screen_candidates.csv         逆向筛选候选
输出：
    demo/molloop_demo2.html
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo"

CLASSIC = {
    "c1ccccc1": "苯", "Cc1ccccc1": "甲苯", "c1ccc2ccccc2c1": "萘",
    "c1ccc2c(c1)ccc3ccccc23": "蒽", "c1ccncc1": "吡啶", "c1ccoc1": "呋喃",
    "c1ccsc1": "噻吩", "c1cc[nH]c1": "吡咯", "Nc1ccccc1": "苯胺",
    "Oc1ccccc1": "苯酚", "CC(=O)O": "乙酸", "CCO": "乙醇", "CCCC": "正丁烷",
    "C#N": "氢氰酸", "O=C=O": "二氧化碳", "C1CC1": "环丙烷", "C1CCC1": "环丁烷",
    "c1ccc2[nH]ccc2c1": "吲哚", "c1cnc2ccccc2c1": "喹啉", "O=C1CCCCC1": "环己酮",
}

UNITS = {"gap": "eV", "homo": "eV", "lumo": "eV", "mu": "Debye", "alpha": "Å³", "cv": "cal/mol·K"}
LABELS = {"gap": "能隙", "homo": "HOMO 能级", "lumo": "LUMO 能级",
          "mu": "偶极矩", "alpha": "极化率", "cv": "热容"}


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
        ["mol_id", "tree_std", "lo90", "hi90", "in_domain", "ad_dist"]]
    df = feat.merge(multi, on="mol_id", how="inner").merge(unc, on="mol_id", how="left")
    m = json.loads((ROOT / "eval" / "metrics.json").read_text(encoding="utf-8"))
    mx = json.loads((ROOT / "eval" / "metrics_extra.json").read_text(encoding="utf-8"))
    mm = json.loads((ROOT / "eval" / "metrics_multi.json").read_text(encoding="utf-8"))
    uj = json.loads((ROOT / "eval" / "uncertainty.json").read_text(encoding="utf-8"))

    # 1) 挑分子：经典分子 + 覆盖各能隙区间
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
        for _, r in sub.sample(min(7, len(sub)), random_state=int(q * 100)).iterrows():
            mol = Chem.MolFromSmiles(r["smiles"])
            picked.append((r["mol_id"], rdMolDescriptors.CalcMolFormula(mol) if mol else ""))
    picked = list(dict.fromkeys(picked))

    cards = []
    for mid, name in picked:
        r = id2.loc[mid]
        mol = Chem.MolFromSmiles(r["smiles"])
        cards.append({
            "id": str(mid), "name": name, "smiles": r["smiles"], "img": img_b64(r["smiles"]),
            "formula": rdMolDescriptors.CalcMolFormula(mol) if mol else "",
            "atoms": int(r["n_atoms"]), "rings": int(r["n_rings"]),
            "pred": {k: round(float(r[f"pred_{k}"]), 3) for k in UNITS},
            "true": {k: round(float(r[f"true_{k}"]), 3) for k in UNITS},
            "std": round(float(r["tree_std"]), 3) if pd.notna(r["tree_std"]) else None,
            "lo90": round(float(r["lo90"]), 3) if pd.notna(r["lo90"]) else None,
            "hi90": round(float(r["hi90"]), 3) if pd.notna(r["hi90"]) else None,
            "in_domain": bool(r["in_domain"]) if pd.notna(r["in_domain"]) else True,
        })
    print(f"[demo2] 分子 {len(cards)} 个", flush=True)

    # 2) 每个性质各抽 3000 个点画散点
    scatter = {}
    for k in UNITS:
        s = df[[f"true_{k}", f"pred_{k}"]].dropna().sample(min(3000, len(df)), random_state=42)
        scatter[k] = s.round(2).values.tolist()

    # 3) 逆向筛选候选
    cands = []
    sc_path = ROOT / "eval" / "screen_candidates.csv"
    if sc_path.exists():
        sc = pd.read_csv(sc_path)
        for _, r in sc.iterrows():
            cands.append({
                "smiles": r["smiles"], "img": img_b64(r["smiles"], 160),
                "pred": round(float(r["pred_gap_eV"]), 3),
                "lo": float(str(r["pred_q90_range_eV"]).split("~")[0]),
                "hi": float(str(r["pred_q90_range_eV"]).split("~")[1]),
                "std": round(float(r["tree_std_eV"]), 3),
                "true": round(float(r["dft_gap_eV"]), 3),
                "err": round(float(r["abs_err_eV"]), 3),
                "atoms": int(r["n_atoms"]), "rings": int(r["n_rings"]),
                "scaffold": r["scaffold"],
            })
    sm = json.loads((ROOT / "eval" / "screen_meta.json").read_text(encoding="utf-8")) \
        if (ROOT / "eval" / "screen_meta.json").exists() else {"window_eV": [5.5, 6.5]}

    data = {
        "cards": cards, "scatter": scatter, "cands": cands,
        "units": UNITS, "labels": LABELS,
        "metrics": {
            "multi": {k: {"mae": mm[k]["mae"], "r2": mm[k]["r2"], "unit": mm[k]["unit"]} for k in UNITS},
            "mae_scaffold": mx["rf_scaffold"]["mae"], "r2_scaffold": mx["rf_scaffold"]["r2"],
            "dummy": mx["dummy_mean"]["mae"], "ridge": mx["ridge"]["mae"],
            "consistency": mm["consistency"],
            "unc": {
                "q90": uj["conformal"]["q90"]["half_width"], "cov90": uj["conformal"]["q90"]["coverage"],
                "q95": uj["conformal"]["q95"]["half_width"],
                "thr": uj["applicability_domain"]["threshold"],
                "in_ratio": uj["applicability_domain"]["in_domain_ratio"],
                "mae_in": uj["applicability_domain"]["mae_in_domain"],
                "mae_out": uj["applicability_domain"]["mae_out_of_domain"],
                "rho": uj["tree_std_vs_error_spearman"],
            },
        },
        "screen_meta": sm,
    }

    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    path = OUT / "molloop_demo2.html"
    path.write_text(html, encoding="utf-8")
    print(f"[demo2] 写出 {path} ({path.stat().st_size/1024:.0f} KB)", flush=True)


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>MolLoop 2.0 · 分子性质预测与逆向筛选</title>
<style>
:root{--blue:#1a6fbf;--blue-d:#0f4c86;--blue-l:#e8f2fb;--ink:#16283d;--gray:#6b7b8c;--line:#d8e3ee}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:"Microsoft YaHei","Segoe UI",sans-serif;background:#f5f9fd;color:var(--ink)}
.wrap{max-width:1220px;margin:0 auto;padding:24px 20px 56px}
header{display:flex;align-items:flex-end;justify-content:space-between;border-bottom:2px solid var(--blue);padding-bottom:13px;margin-bottom:20px}
h1{font-size:25px;color:var(--blue-d)} h1 span{color:var(--blue)}
.sub{font-size:13px;color:var(--gray);margin-top:6px}
.tag{background:var(--blue);color:#fff;font-size:12px;padding:5px 12px;border-radius:3px}
.kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:20px}
.kpi{background:#fff;border:1px solid var(--line);border-radius:6px;padding:12px 13px}
.kpi .k{font-size:11px;color:var(--gray);margin-bottom:7px;white-space:nowrap}
.kpi .v{font-size:19px;font-weight:700;color:var(--blue-d)}
.kpi .v small{font-size:11px;font-weight:400;color:var(--gray);margin-left:3px}
.tabs{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}
.tab{border:1px solid var(--line);background:#fff;border-radius:4px;padding:7px 14px;font-size:13px;cursor:pointer;color:var(--gray)}
.tab.on{background:var(--blue);border-color:var(--blue);color:#fff}
.main{display:grid;grid-template-columns:1fr 400px;gap:18px}
.panel{background:#fff;border:1px solid var(--line);border-radius:6px;padding:15px}
.panel h2{font-size:15px;color:var(--blue-d);margin-bottom:11px;padding-left:9px;border-left:3px solid var(--blue)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(100px,1fr));gap:9px;max-height:540px;overflow:auto;padding-right:4px}
.card{border:1px solid var(--line);border-radius:5px;padding:5px;text-align:center;cursor:pointer;transition:.15s;background:#fff}
.card:hover{border-color:var(--blue);box-shadow:0 2px 8px rgba(26,111,191,.15)}
.card.on{border-color:var(--blue);background:var(--blue-l)}
.card img{width:100%;height:66px;object-fit:contain}
.card .nm{font-size:11px;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card .pv{font-size:11px;color:var(--blue);font-weight:700}
.detail .big{font-size:42px;font-weight:800;color:var(--blue);line-height:1.08;margin:2px 0 0}
.detail .big small{font-size:14px;color:var(--gray);font-weight:400}
.detail .pm{font-size:13px;color:var(--gray);margin-bottom:8px}
.detail img{width:180px;display:block;margin:0 auto 8px}
.badge{display:inline-block;font-size:11px;padding:3px 9px;border-radius:10px;margin-right:6px}
.badge.ok{background:#e6f6ed;color:#1a9a5b} .badge.warn{background:#fdf1e3;color:#c0691a}
.rows{font-size:12.5px;margin-top:10px;border-top:1px dashed var(--line);padding-top:9px}
.row{display:flex;justify-content:space-between;padding:4px 0;color:var(--gray);gap:10px}
.row b{color:var(--ink);font-weight:600;text-align:right}
.hint{font-size:12px;color:var(--gray);margin-top:10px;line-height:1.65}
canvas{width:100%;height:300px;display:block}
.screen{margin-top:18px}
.srow{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px;font-size:13px;color:var(--gray)}
.srow input{width:76px;padding:5px 8px;border:1px solid var(--line);border-radius:4px;font-size:13px;color:var(--ink)}
.srow button{background:var(--blue);color:#fff;border:none;border-radius:4px;padding:6px 14px;font-size:13px;cursor:pointer}
.tbl{width:100%;border-collapse:collapse;font-size:12.5px}
.tbl th{background:var(--blue-l);color:var(--blue-d);font-weight:600;padding:7px 6px;text-align:left;border-bottom:1px solid var(--line)}
.tbl td{padding:6px;border-bottom:1px solid #eef4fa;vertical-align:middle}
.tbl img{height:46px;display:block}
.tbl .sm{font-size:11px;color:var(--gray);word-break:break-all}
.smeta{font-size:12px;color:var(--gray);margin-bottom:10px;line-height:1.7}
footer{margin-top:22px;border-top:1px solid var(--line);padding-top:12px;font-size:12px;color:var(--gray);line-height:1.9}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>Mol<span>Loop</span> 2.0 · 分子性质预测与逆向筛选</h1>
      <p class="sub">同一套 217 维 2D 描述符，同时预测 6 个量子化学性质；带不确定度、适用域判断和逆向筛选。全部离线运行。</p>
    </div>
    <div class="tag">QM9 · 133,802 个分子</div>
  </header>

  <div class="kpis" id="kpis"></div>
  <div class="tabs" id="tabs"></div>

  <div class="main">
    <div class="panel">
      <h2>选一个分子（当前性质：<span id="curname">能隙</span>）</h2>
      <div class="grid" id="grid"></div>
      <div class="hint">卡片数字是模型<b>没见过该分子</b>时的留出预测；点开看完整预测区间与适用域判断。</div>
    </div>
    <div class="panel detail">
      <h2>预测详情</h2>
      <div id="detail"></div>
    </div>
  </div>

  <div class="panel" style="margin-top:18px">
    <h2>全数据集表现：预测值 vs DFT 真值（<span id="sname">能隙</span>）</h2>
    <canvas id="cv"></canvas>
    <div class="hint">每点为一个抽样分子；红点为当前选中分子。越贴近虚线越准。</div>
  </div>

  <div class="panel screen">
    <h2>逆向筛选：给定目标能隙窗口，从 13.38 万里挑候选</h2>
    <div class="smeta" id="smeta"></div>
    <div class="srow">
      目标窗口 <input id="wlo" type="number" step="0.1"> – <input id="whi" type="number" step="0.1"> eV
      <button onclick="doScreen()">筛选</button>
      <span id="scount"></span>
    </div>
    <table class="tbl" id="stbl"></table>
  </div>

  <footer>
    模型：RDKit 217 维 2D 描述符 + 随机森林（200 棵树，max_features 0.5）｜交叉验证 5 折留出预测（OOF）<br>
    不确定度＝200 棵树预测值的标准差；区间＝OOF 残差分位数的 conformal 区间；适用域＝PCA(20) 空间的 kNN 距离阈值<br>
    全流程本地 CPU 复现，脚本：train/multi_target.py、eval/uncertainty.py、train/screen.py、demo/build_demo2.py
  </footer>
</div>

<script>
const DATA = __DATA__;
const $ = s => document.querySelector(s);
const U = DATA.units, L = DATA.labels;
let curProp = 'gap', curCard = 0;

// 顶部 6 个性质指标
$('#kpis').innerHTML = Object.keys(U).map(k =>
  '<div class="kpi"><div class="k">' + L[k] + ' MAE</div><div class="v">' +
  DATA.metrics.multi[k].mae.toFixed(3) + '<small>' + U[k] + '</small></div></div>').join('');

// 性质切换
$('#tabs').innerHTML = Object.keys(U).map((k, i) =>
  '<div class="tab' + (i === 0 ? ' on' : '') + '" data-p="' + k + '">' + L[k] + '</div>').join('');
document.querySelectorAll('.tab').forEach(t => t.onclick = () => {
  curProp = t.dataset.p;
  document.querySelectorAll('.tab').forEach(e => e.classList.toggle('on', e.dataset.p === curProp));
  $('#curname').textContent = L[curProp]; $('#sname').textContent = L[curProp];
  renderGrid(); pick(curCard); draw();
});

function renderGrid(){
  $('#grid').innerHTML = DATA.cards.map((c, i) =>
    '<div class="card' + (i === curCard ? ' on' : '') + '" data-i="' + i + '">' +
    '<img src="' + c.img + '"><div class="nm">' + c.name + '</div>' +
    '<div class="pv">' + c.pred[curProp].toFixed(2) + '</div></div>').join('');
  document.querySelectorAll('.card').forEach(d => d.onclick = () => pick(+d.dataset.i));
}

function pick(i){
  curCard = i;
  document.querySelectorAll('.card').forEach(e => e.classList.toggle('on', +e.dataset.i === i));
  const c = DATA.cards[i], p = curProp, unit = U[p];
  const pr = c.pred[p], tr = c.true[p], ae = Math.abs(pr - tr);
  let interval = '', badge = '';
  if (p === 'gap' && c.lo90 !== null){
    interval = '<div class="pm">90% 区间 ' + c.lo90.toFixed(2) + ' ~ ' + c.hi90.toFixed(2) +
               ' eV（半宽 ±' + DATA.metrics.unc.q90.toFixed(2) + '）｜树间标准差 ' + c.std.toFixed(3) + '</div>';
    badge = c.in_domain
      ? '<span class="badge ok">在适用域内</span>'
      : '<span class="badge warn">超出适用域 · 预测仅供参考</span>';
  }
  const errCls = ae <= 0.2 ? 'good' : 'warn';
  $('#detail').innerHTML =
    '<img src="' + c.img + '">' +
    '<div class="big">' + pr.toFixed(2) + '<small> ' + unit + ' 预测</small></div>' +
    interval + badge +
    '<div class="rows">' +
      '<div class="row"><span>分子</span><b>' + c.name + '</b></div>' +
      '<div class="row"><span>分子式</span><b>' + c.formula + '</b></div>' +
      '<div class="row"><span>SMILES</span><b style="font-size:10.5px">' + c.smiles + '</b></div>' +
      '<div class="row"><span>DFT 真值</span><b>' + tr.toFixed(2) + ' ' + unit + '</b></div>' +
      '<div class="row"><span>绝对误差</span><b style="color:' + (ae <= 0.2 ? '#1a9a5b' : '#c0691a') + '">' + ae.toFixed(3) + '</b></div>' +
      '<div class="row"><span>重原子 / 环数</span><b>' + c.atoms + ' / ' + c.rings + '</b></div>' +
      '<div class="row"><span>一致性检验</span><b>gap ≈ LUMO − HOMO（两条路线差 ' +
        DATA.metrics.consistency.homo_lumo_diff_mae.toFixed(3) + ' eV）</b></div>' +
    '</div>' +
    '<div class="hint">这个分子上，模型给出 ' + pr.toFixed(2) + ' ± ' + (c.std === null ? '—' : c.std.toFixed(2)) +
    ' ' + unit + '；不确定度越大越不该直接采信。</div>';
  draw();
}

const cv = $('#cv'), ctx = cv.getContext('2d');
function draw(){
  const W = cv.clientWidth, H = cv.clientHeight, P = 40;
  const dpr = window.devicePixelRatio || 1;
  cv.width = W * dpr; cv.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
  const pts = DATA.scatter[curProp] || [];
  const all = pts.map(q => [q[0], q[1]]);
  const c = DATA.cards[curCard];
  if (c) all.push([c.true[curProp], c.pred[curProp]]);
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
  ctx.beginPath(); ctx.moveTo(X(x0), Y(x0)); ctx.lineTo(X(x1), Y(x1)); ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = 'rgba(26,111,191,.30)';
  pts.forEach(q => { ctx.beginPath(); ctx.arc(X(q[0]), Y(q[1]), 1.5, 0, 6.2832); ctx.fill(); });
  if (c){
    ctx.beginPath(); ctx.arc(X(c.true[curProp]), Y(c.pred[curProp]), 6, 0, 6.2832);
    ctx.fillStyle = '#d0342c'; ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
  }
  ctx.fillStyle = '#6b7b8c'; ctx.font = '12px "Microsoft YaHei"';
  ctx.fillText('DFT 真值 (' + U[curProp] + ')', W / 2 - 46, H - 10);
  ctx.save(); ctx.translate(12, H / 2 + 40); ctx.rotate(-Math.PI / 2);
  ctx.fillText('模型预测', 0, 0); ctx.restore();
}

// 逆向筛选
$('#wlo').value = DATA.screen_meta.window_eV[0];
$('#whi').value = DATA.screen_meta.window_eV[1];
$('#smeta').innerHTML = '流程：预测落窗 → 适用域过滤 → 树间标准差排序 → 骨架去重（每种 Bemis-Murcko 骨架只留 1 个）' +
  '<br>当前候选库共 ' + DATA.cands.length + ' 个，覆盖 ' + new Set(DATA.cands.map(c => c.scaffold)).size + ' 个不同骨架。';
function doScreen(){
  const lo = +$('#wlo').value, hi = +$('#whi').value;
  const hit = DATA.cands.filter(c => c.pred >= lo && c.pred <= hi);
  $('#scount').textContent = '命中 ' + hit.length + ' 个';
  $('#stbl').innerHTML = '<tr><th>结构</th><th>SMILES</th><th>预测能隙</th><th>90% 区间</th><th>树间标准差</th><th>DFT 真值</th><th>误差</th><th>原子/环</th></tr>' +
    hit.map(c => '<tr><td><img src="' + c.img + '"></td><td class="sm">' + c.smiles + '</td>' +
      '<td><b>' + c.pred.toFixed(2) + '</b></td><td>' + c.lo.toFixed(2) + '~' + c.hi.toFixed(2) + '</td>' +
      '<td>' + c.std.toFixed(3) + '</td><td>' + c.true.toFixed(2) + '</td>' +
      '<td style="color:' + (c.err <= 0.2 ? '#1a9a5b' : '#c0691a') + '">' + c.err.toFixed(3) + '</td>' +
      '<td>' + c.atoms + ' / ' + c.rings + '</td></tr>').join('');
}
doScreen();

renderGrid(); pick(0);
window.addEventListener('resize', () => draw());
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()

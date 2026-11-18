# -*- coding: utf-8 -*-
"""把 Agent 自循环的 jsonl 留痕渲染成一张离线决策时间线页面。

用法： python demo/render_trace.py [trace 文件名或路径]
输出： demo/agent_timeline.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs"


def pick_trace() -> Path:
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        return p if p.is_absolute() else LOG / p
    files = sorted(LOG.glob("trace_*.jsonl"))
    return files[-1]


def main() -> None:
    tp = pick_trace()
    steps = [json.loads(l) for l in tp.read_text(encoding="utf-8").splitlines() if l.strip()]
    run = steps[0]["run_id"]
    baseline = next((s["payload"].get("mae") for s in steps if s["step"] == "baseline"), None)
    end = next((s["payload"] for s in steps if s["step"] == "run_end"), {})
    best = end.get("best_mae")
    rounds = end.get("rounds", 0)
    total = steps[-1]["elapsed"]

    cards = []
    for s in steps:
        kind, p, ts, el = s["step"], s["payload"], s["ts"], s["elapsed"]
        if kind == "proposal":
            plan = p.get("plan", {})
            cards.append({
                "t": ts[11:19], "el": el, "round": p.get("round"),
                "h": plan.get("hypothesis", ""),
                "changes": plan.get("changes", {}),
                "expect": plan.get("expect", ""),
                "rollback": plan.get("rollback_if", ""),
            })
        elif kind == "apply_config":
            cards.append({"t": ts[11:19], "el": el, "round": p.get("round"), "apply": p.get("changes", {})})
        elif kind == "train_run":
            tail = (p.get("tail") or "").strip().splitlines()
            cards.append({"t": ts[11:19], "el": el, "round": p.get("round"),
                          "train": {"exit": p.get("exit_code"), "tail": tail[-1] if tail else ""}})
        elif kind == "evaluate":
            cards.append({"t": ts[11:19], "el": el, "round": p.get("round"),
                          "eval": {"before": p.get("before"), "after": p.get("after"), "delta": p.get("delta")}})

    data = {"run": run, "baseline": baseline, "best": best, "rounds": rounds,
            "total": round(total, 1), "cards": cards, "n_steps": len(steps)}

    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    out = ROOT / "demo" / "agent_timeline.html"
    out.write_text(html, encoding="utf-8")
    print(f"[render] {tp.name} -> {out} ({out.stat().st_size/1024:.0f} KB)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>MolLoop Agent · 自循环决策留痕</title>
<style>
:root{--blue:#1a6fbf;--blue-d:#0f4c86;--blue-l:#e8f2fb;--ink:#16283d;--gray:#6b7b8c;--line:#d8e3ee}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:"Microsoft YaHei","Segoe UI",sans-serif;background:#f5f9fd;color:var(--ink)}
.wrap{max-width:1000px;margin:0 auto;padding:24px 20px 60px}
header{border-bottom:2px solid var(--blue);padding-bottom:13px;margin-bottom:20px;display:flex;justify-content:space-between;align-items:flex-end}
h1{font-size:24px;color:var(--blue-d)} h1 span{color:var(--blue)}
.sub{font-size:13px;color:var(--gray);margin-top:6px}
.tag{background:var(--blue);color:#fff;font-size:12px;padding:5px 12px;border-radius:3px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:22px}
.kpi{background:#fff;border:1px solid var(--line);border-radius:6px;padding:13px 15px}
.kpi .k{font-size:11px;color:var(--gray);margin-bottom:6px}
.kpi .v{font-size:20px;font-weight:700;color:var(--blue-d)}
.kpi .v small{font-size:11px;color:var(--gray);font-weight:400;margin-left:3px}
.tl{position:relative;padding-left:26px}
.tl:before{content:"";position:absolute;left:7px;top:6px;bottom:6px;width:2px;background:var(--line)}
.item{position:relative;margin-bottom:14px}
.dot{position:absolute;left:-24px;top:16px;width:11px;height:11px;border-radius:50%;background:var(--blue);border:2px solid #fff;box-shadow:0 0 0 2px var(--blue-l)}
.dot.ev{background:#1a9a5b;box-shadow:0 0 0 2px #e6f6ed}
.dot.run{background:#c0691a;box-shadow:0 0 0 2px #fdf1e3}
.card{background:#fff;border:1px solid var(--line);border-radius:6px;padding:13px 15px}
.card .hd{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.card .ttl{font-size:14px;font-weight:700;color:var(--blue-d)}
.card .tm{font-size:11px;color:var(--gray)}
.lbl{font-size:11px;color:#fff;background:var(--blue);padding:2px 8px;border-radius:9px;margin-left:7px}
.lbl.g{background:#1a9a5b} .lbl.o{background:#c0691a}
.field{font-size:12.5px;color:var(--gray);line-height:1.75;margin-top:6px}
.field b{color:var(--ink)}
.field .big{color:var(--blue);font-weight:700}
.chip{display:inline-block;background:var(--blue-l);color:var(--blue-d);font-size:11.5px;padding:2px 8px;border-radius:4px;margin:2px 4px 2px 0;font-family:Consolas,monospace}
footer{margin-top:22px;border-top:1px solid var(--line);padding-top:12px;font-size:12px;color:var(--gray);line-height:1.9}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>Mol<span>Loop</span> · Agent 自循环决策留痕</h1>
      <p class="sub">大模型读指标 → 提假设 → 改配置 → 触发训练 → 读新指标 → 判断采纳或回滚。下面是真实运行记录，逐条来自 JSONL 留痕。</p>
    </div>
    <div class="tag" id="run"></div>
  </header>
  <div class="kpis" id="kpis"></div>
  <div class="tl" id="tl"></div>
  <footer>
    数据来源：logs/<span id="run2"></span>.jsonl｜每一步带时间戳与累计耗时，可逐条复核<br>
    回滚机制：若该轮指标未改善，配置自动恢复到上一轮快照（logs/config_snapshots/）
  </footer>
</div>
<script>
const DATA = __DATA__;
const $ = s => document.querySelector(s);
$('#run').textContent = 'run ' + DATA.run;
$('#run2').textContent = DATA.run;
$('#kpis').innerHTML =
  '<div class="kpi"><div class="k">起始 MAE</div><div class="v">' + DATA.baseline.toFixed(4) + '<small>eV</small></div></div>' +
  '<div class="kpi"><div class="k">最终 MAE</div><div class="v">' + DATA.best.toFixed(4) + '<small>eV</small></div></div>' +
  '<div class="kpi"><div class="k">采纳轮次</div><div class="v">' + DATA.rounds + '<small>轮</small></div></div>' +
  '<div class="kpi"><div class="k">总耗时</div><div class="v">' + (DATA.total/60).toFixed(1) + '<small>分钟</small></div></div>';

function chip(o){
  return Object.entries(o).map(([k,v]) => '<span class="chip">' + k + ' = ' + v + '</span>').join('');
}
$('#tl').innerHTML = DATA.cards.map(c => {
  if (c.h !== undefined){
    return '<div class="item"><div class="dot"></div><div class="card">' +
      '<div class="hd"><span class="ttl">第 ' + c.round + ' 轮假设<span class="lbl">大模型提出</span></span><span class="tm">' + c.t + ' · 累计 ' + c.el.toFixed(0) + 's</span></div>' +
      '<div class="field"><b>假设：</b>' + c.h + '</div>' +
      '<div class="field"><b>改动：</b>' + chip(c.changes) + '</div>' +
      '<div class="field"><b>预期与判据：</b>' + c.expect + '</div>' +
      '<div class="field"><b>回滚条件：</b>' + c.rollback + '</div>' +
      '</div></div>';
  }
  if (c.apply !== undefined){
    return '<div class="item"><div class="dot run"></div><div class="card">' +
      '<div class="hd"><span class="ttl">写入配置文件<span class="lbl o">自动执行</span></span><span class="tm">' + c.t + '</span></div>' +
      '<div class="field">configs/base.json ← ' + chip(c.apply) + '</div></div></div>';
  }
  if (c.train !== undefined){
    return '<div class="item"><div class="dot run"></div><div class="card">' +
      '<div class="hd"><span class="ttl">触发重训<span class="lbl o">自动执行</span></span><span class="tm">' + c.t + '</span></div>' +
      '<div class="field">退出码 ' + c.train.exit + '｜' + (c.train.tail || '') + '</div></div></div>';
  }
  if (c.eval !== undefined){
    const d = c.eval.delta;
    return '<div class="item"><div class="dot ev"></div><div class="card">' +
      '<div class="hd"><span class="ttl">复评：' + c.eval.before.toFixed(4) + ' → ' + c.eval.after.toFixed(4) + '<span class="lbl g">' +
      (d < 0 ? '采纳' : '回滚') + '</span></span><span class="tm">' + c.t + '</span></div>' +
      '<div class="field">ΔMAE = <span class="big">' + d.toFixed(4) + ' eV</span>（' + (d < 0 ? '改善' : '未改善') + '）</div></div></div>';
  }
  return '';
}).join('');
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""让大模型读实验数据，自动写"结果与讨论"章节。

约束：
- 所有数字只能来自给定 JSON，禁止编造；
- 必须点出最大风险与改进方向；
- 输出中文 Markdown，直接可放进方案书。

输出：eval/ai_conclusion.md（并在 logs/ 记录 token 用量）
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"

SYSTEM = """你是计算化学与机器学习交叉方向的研究者，正在把实验数据整理成论文/方案书的"结果与讨论"章节。
要求：
1. 只能使用我给你的数字，任何数字都要能在给定 JSON 里找到，禁止自行推测或补造；
2. 结论要具体，不写"效果良好""意义重大"这类空话，每条结论后面必须跟一个具体数值；
3. 必须指出这份工作最大的一个风险，以及两条下一步改进方向；
4. 语气客观克制，承认与主流方法（图神经网络类）的差距，不做夸大；
5. 输出中文 Markdown，结构：## 核心结论 / ## 与文献的定位 / ## 局限性 / ## 下一步；总长度 600-900 字。"""


def collect() -> dict:
    out = {}
    for name in ("metrics.json", "metrics_extra.json", "metrics_multi.json",
                 "uncertainty.json", "screen_meta.json"):
        p = EVAL / name
        if p.exists():
            out[name] = json.loads(p.read_text(encoding="utf-8"))
    return out


def main() -> None:
    cfg = json.loads((ROOT / "configs" / "llm.json").read_text(encoding="utf-8"))
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg.get("base_url"))
    data = collect()

    t0 = time.time()
    r = client.chat.completions.create(
        model=cfg.get("model"),
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "实验数据如下（JSON）：\n" + json.dumps(data, ensure_ascii=False, indent=1)
             + "\n\n请按系统要求输出 Markdown 正文，不要输出多余解释。"},
        ],
        temperature=1,   # 该模型强制 1
    )
    txt = (r.choices[0].message.content or "").strip()
    usage = r.usage

    md = f"""# AI 自动生成的"结果与讨论"

> 由大模型（{cfg.get('model')}）读取 `eval/` 下全部指标文件后自动撰写，
> 生成时间 {time.strftime('%Y-%m-%d %H:%M:%S')}，耗时 {time.time()-t0:.0f} 秒，
> token 用量 {usage.prompt_tokens} 输入 / {usage.completion_tokens} 输出。
> 这是"AI 参与科研流程"的直接证据：不仅调超参，还参与结果解读。

{txt}
"""
    (EVAL / "ai_conclusion.md").write_text(md, encoding="utf-8")
    print(f"[writer] 写出 {EVAL / 'ai_conclusion.md'} | {len(txt)} 字 | token {usage.total_tokens}", flush=True)


if __name__ == "__main__":
    main()

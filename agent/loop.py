# -*- coding: utf-8 -*-
"""自我循环（Self-Loop）：执行 -> 验证 -> 归因 -> 修正，最多 3 轮，逐轮留痕。

用法：
    python agent/loop.py --model rf --rounds 3
需要环境变量 LLM_API_KEY（DeepSeek / 通义 / 任意 OpenAI 兼容接口）。
未配置 Key 时进入 dry-run：只记录当前指标与建议位，不调用大模型。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace import Tracer  # noqa: E402
import prompts as P  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "base.json"
EVAL = ROOT / "eval" / "metrics.json"


def read_metrics(model: str) -> dict:
    if not EVAL.exists():
        return {}
    return json.loads(EVAL.read_text(encoding="utf-8")).get(model, {})


def read_top_features(model: str, k: int = 10) -> list[str]:
    f = ROOT / "train" / f"importance_{model}.csv"
    if not f.exists():
        return []
    return pd.read_csv(f).head(k)["feature"].tolist()


def llm_conf() -> dict:
    """优先读环境变量，其次读 configs/llm.json（方便队友不配环境变量也能跑）。"""
    keys = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
    conf = {k: os.getenv(k) for k in keys}
    if conf["LLM_API_KEY"]:
        return conf
    f = ROOT / "configs" / "llm.json"
    if f.exists():
        c = json.loads(f.read_text(encoding="utf-8"))
        return {
            "LLM_API_KEY": c.get("api_key"),
            "LLM_BASE_URL": c.get("base_url", "https://ark.cn-beijing.volces.com/api/v3"),
            "LLM_MODEL": c.get("model", "kimi-k3-1-pro-261024"),
        }
    return conf


def ask_llm(system: str, user: str) -> dict | None:
    """调用 OpenAI 兼容接口；未配置 Key 返回 None，触发 dry-run。"""
    conf = llm_conf()
    key = conf.get("LLM_API_KEY")
    if not key or key.startswith("在这里"):
        return None
    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=conf.get("LLM_BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3")
    resp = client.chat.completions.create(
        model=conf.get("LLM_MODEL") or "kimi-k3-1-pro-261024",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=1,  # 方舟上的 kimi 系列只允许 temperature=1，写死避免 400
    )
    text = resp.choices[0].message.content.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rf")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--compile", default="py_compile")
    args = ap.parse_args()

    tr = Tracer(stage="self_loop")
    history: list[dict] = []
    best = read_metrics(args.model).get("MAE")
    tr.log("baseline", model=args.model, mae=best)

    for r in range(1, args.rounds + 1):
        metrics = read_metrics(args.model)
        if not metrics:
            tr.finish("failed", reason="无 baseline 指标，先跑 eval/metrics.py")
            print("[loop] 缺少 baseline 指标，终止")
            return

        cfg = json.loads(CFG.read_text(encoding="utf-8"))
        plan = ask_llm(P.SYSTEM, P.loop_prompt(metrics, cfg, history))

        if plan is None:
            tr.log("dry_run", round=r, note="未配置 LLM_API_KEY，仅记录当前状态",
                   metrics=metrics, top_features=read_top_features(args.model))
            print(f"[loop] round {r}: dry-run（未配置 LLM_API_KEY），当前 MAE={metrics.get('MAE')}")
            break

        tr.log("proposal", round=r, plan=plan)
        changes = plan.get("changes", {}) if isinstance(plan, dict) else {}
        if not changes:
            tr.log("no_change", round=r)
            break

        Tracer.snapshot_config(CFG, f"r{r}_before")
        for k, v in changes.items():
            if "." in k:
                group, key = k.split(".", 1)
                cfg.setdefault(group, {})[key] = v
            else:
                cfg[k] = v
        CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        tr.log("apply_config", round=r, changes=changes)

        code, out = Tracer.run_cmd(["train/baseline.py", "--model", args.model])
        tr.log("train_run", round=r, exit_code=code, tail=out)
        if code != 0:
            tr.log("rollback", round=r, reason="训练失败")
            break
        Tracer.run_cmd(["eval/metrics.py", "--models", args.model])

        new_mae = read_metrics(args.model).get("MAE")
        delta = None if (best is None or new_mae is None) else round(new_mae - best, 4)
        tr.log("evaluate", round=r, before=best, after=new_mae, delta=delta)
        print(f"[loop] round {r}: MAE {best} -> {new_mae} (delta={delta})")
        history.append({"round": r, "changes": changes, "before": best, "after": new_mae})

        if delta is not None and delta < 0:
            best = new_mae
        else:
            # 未改善则回滚，避免越调越差
            for p in sorted((ROOT / "logs" / "config_snapshots").glob(f"r{r}_before_*.json")):
                CFG.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
            tr.log("rollback", round=r, reason="本轮未改善，配置已回滚")
            break

    tr.finish("ok", rounds=len(history), best_mae=best)
    print(f"[loop] 结束，最佳 MAE={best}，留痕文件 {tr.path}")


if __name__ == "__main__":
    main()

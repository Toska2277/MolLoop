# -*- coding: utf-8 -*-
"""调用方舟接口，为分子生成中文/英文系统命名，并做一致性校验。

校验方式：让模型同时回填 SMILES，与输入比对；RDKit 再核一遍分子式。
对不上的丢弃不采用，绝不把没把握的名字写进页面。
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from openai import OpenAI
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "demo" / "iupac_cache.json"
sys.path.insert(0, str(ROOT / "demo"))

CFG = ROOT / "configs" / "llm.json"
PROMPT = """你是 IUPAC 有机化学命名专家。请对下面这个分子给出命名。

SMILES: {smiles}

要求：
1. 给出中文 IUPAC 系统命名（简体中文，用“-”连位次，例如“1,3-苯二酚”不用，请用“间苯二酚”这类系统名时写标准中文系统命名，如“1,3-二羟基苯”）。
2. 给出英文 IUPAC 名。
3. 回填你据以命名的 SMILES（必须与给定 SMILES 表示同一分子）。
4. 只输出一行 JSON，不要任何解释：{{"zh":"...","en":"...","echo":"..."}}"""


def client():
    c = json.loads(CFG.read_text(encoding="utf-8"))
    return (OpenAI(api_key=c["api_key"], base_url=c["base_url"], max_retries=1, timeout=240),
            c.get("model", "deepseek-v4-1-flash-economy-261017"))


def same(a: str, b: str) -> bool:
    ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    if ma is None or mb is None:
        return False
    try:
        from rdkit.Chem import inchi
        if inchi.MolToInchiKey(ma) == inchi.MolToInchiKey(mb):
            return True
    except Exception:  # noqa: BLE001
        pass
    return Chem.MolToSmiles(ma) == Chem.MolToSmiles(mb)


def one(cli, model, smi):
    try:
        r = cli.chat.completions.create(
            model=model, temperature=1,
            messages=[{"role": "user", "content": PROMPT.format(smiles=smi)}],
            timeout=150,
        )
        txt = r.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", txt, re.S)
        obj = json.loads(m.group(0)) if m else {}
    except Exception as e:  # noqa: BLE001
        return smi, None, f"失败 {e}"
    zh, en, echo = obj.get("zh", "").strip(), obj.get("en", "").strip(), obj.get("echo", "")
    if not zh or not echo or not same(smi, echo):
        return smi, None, f"校验不通过 {zh}"
    mol = Chem.MolFromSmiles(smi)
    return smi, {"zh": zh, "en": en, "f": rdMolDescriptors.CalcMolFormula(mol)}, ""


def main() -> None:
    from concurrent.futures import ThreadPoolExecutor

    targets = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if isinstance(targets, dict):
        targets = list(targets.keys())
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    cli, model = client()
    todo = [s for s in targets if s not in cache]
    print(f"[name] 待生成 {len(todo)} 个，已有 {len(cache)} 个", flush=True)

    done = [0]
    def task(smi):
        smi, res, msg = one(cli, model, smi)
        if res:
            cache[smi] = res
        done[0] += 1
        print(f"[name] {done[0]}/{len(todo)} {'✓' if res else '✗'} {smi} -> {res['zh'] if res else msg}", flush=True)
        if done[0] % 5 == 0:
            CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(task, todo))
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[name] 写入 {CACHE}，共 {len(cache)} 条", flush=True)


if __name__ == "__main__":
    main()

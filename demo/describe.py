# -*- coding: utf-8 -*-
"""把分子结构翻译成中文片段描述（苯环·羟基·甲基 这种），存成短代码塞进页面。

说明：这不是 IUPAC 系统命名。离线批量做"结构 -> IUPAC 中文名"没有可靠开源方案
（OPSIN 只能名字->结构，反向要商业软件），所以这里给的是**功能团结构描述**，
够看懂分子长什么样，且不编造名字。
"""
from __future__ import annotations

from rdkit import Chem

# 代码, 中文, SMARTS
PATTERNS = [
    ("B", "苯环", "c1ccccc1"),
    ("b", "萘环", "c1ccc2ccccc2c1"),
    ("I", "吡啶环", "c1ccncc1"),
    ("i", "嘧啶环", "c1cncnc1"),
    ("u", "呋喃环", "c1ccoc1"),
    ("t", "噻吩环", "c1ccsc1"),
    ("p", "吡咯环", "c1cc[nH]c1"),
    ("m", "咪唑环", "c1cnc[nH]1"),
    ("r", "三元环", "C1CC1"),
    ("R", "四元环", "C1CCC1"),
    ("y", "五元环", "C1CCCC1"),
    ("z", "六元环", "C1CCCCC1"),
    ("O", "羟基", "[OX2H]"),
    ("P", "酚羟基", "[OX2H][c]"),
    ("C", "羧基", "[CX3](=O)[OX2H1]"),
    ("E", "酯基", "[CX3](=O)[OX2H0][#6]"),
    ("L", "醛基", "[CH1](=O)[#6]"),
    ("K", "羰基", "[#6][CX3]=[OX1]"),
    ("M", "酰胺", "[n,NX3][c,CX3]=[OX1]"),
    ("N", "氨基", "[NX3H2][#6]"),
    ("Q", "氰基", "[#6]C#N"),
    ("S", "硝基", "[N+](=O)[O-]"),
    ("X", "醚键", "[#6][OX2][#6]"),
    ("V", "碳碳双键", "[#6]=[#6]"),
    ("W", "碳碳三键", "[#6]#[#6]"),
    ("F", "氟", "[F]"),
    ("g", "甲基", "[CH3]"),
    ("h", "乙基", "[CH2][CH3]"),
]

_COMPILED = [(code, cn, Chem.MolFromSmarts(sma)) for code, cn, sma in PATTERNS]
CODE_NAMES = {code: cn for code, cn, _ in PATTERNS}

# 去掉互相包含的重复描述：有了细分项就不再报上位词
_DROP_IF = [
    ("O", {"P", "C"}),   # 有酚羟基或羧基就不再单独说羟基
    ("K", {"C", "E", "M", "L"}),  # 有羧基/酯/酰胺/醛就不说羰基
    ("X", {"E", "M"}),   # 酯基/酰胺里的氧不是醚键
    ("g", {"h"}),        # 有乙基就不说甲基
    ("B", {"b"}),        # 有萘环就不说苯环
]


def describe(mol) -> str:
    """返回短代码串，例如 'BOg' -> 苯环·羟基·甲基。"""
    if mol is None:
        return ""
    out = []
    for code, _cn, patt in _COMPILED:
        if patt is not None and mol.HasSubstructMatch(patt):
            out.append(code)
    present = set(out)
    for code, others in _DROP_IF:
        if present & others and code in present:
            present.discard(code)
    return "".join(c for c in out if c in present)


def describe_text(codes: str) -> str:
    return "·".join(CODE_NAMES.get(c, c) for c in codes) if codes else "结构简单"

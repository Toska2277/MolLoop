# -*- coding: utf-8 -*-
"""自动生成中文系统名（简化规则版）。

覆盖范围有限但保证不硬编错名字：
- 单环（苯 / 吡啶 / 嘧啶 / 吡嗪 / 呋喃 / 噻吩 / 吡咯 / 咪唑 / 脂环）+ 简单取代基
- 直链 C1–C4 的 烷/烯/醇/酮/酸/胺
超出范围返回空串，由调用方退回“结构特征”描述。
"""
from __future__ import annotations

from rdkit import Chem

ALKANE = {1: "甲烷", 2: "乙烷", 3: "丙烷", 4: "丁烷", 5: "戊烷", 6: "己烷"}
ALKENE = {2: "乙烯", 3: "丙烯", 4: "丁烯", 5: "戊烯", 6: "己烯"}


def _chain_name(mol):
    """直链小分子：烷/烯/醇/酮/酸/胺。支链或超过 4 个碳的不猜。"""
    carbons = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C"]
    if not carbons or len(carbons) > 4:
        return ""
    if any(sum(1 for nb in mol.GetAtomWithIdx(i).GetNeighbors() if nb.GetSymbol() == "C") > 2
           for i in carbons):
        return ""
    n = len(carbons)
    h_oh = mol.HasSubstructMatch(Chem.MolFromSmarts("[OX2H1]"))
    h_acid = mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OX2H1]"))
    h_ket = mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3]=O"))
    h_nh2 = mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3H2]"))
    h_ene = mol.HasSubstructMatch(Chem.MolFromSmarts("C=C"))
    if h_acid:
        return {1: "甲酸", 2: "乙酸", 3: "丙酸", 4: "丁酸"}.get(n, "")
    if h_ket and not h_oh:
        if n == 1:
            return "甲醛"
        if n == 2:
            return "乙醛" if mol.HasSubstructMatch(Chem.MolFromSmarts("[CH1]=O")) else "乙醛"
        return {3: "丙酮", 4: "丁酮"}.get(n, "")
    if h_oh:
        return {1: "甲醇", 2: "乙醇", 3: "丙醇", 4: "丁醇"}.get(n, "")
    if h_nh2:
        return {1: "甲胺", 2: "乙胺", 3: "丙胺", 4: "丁胺"}.get(n, "")
    if h_ene:
        return ALKENE.get(n, "")
    return ALKANE.get(n, "")

# 取代基：(名字, SMARTS, 是否需要位号)；顺序即优先级，先命中先用
SUB_PATTERNS = [
    ("羧基", "[a][CX3](=O)[OX2H1]", True),
    ("甲氧基", "[a][OX2][CH3]", True),
    ("乙氧基", "[a][OX2][CH2][CH3]", True),
    ("羟基", "[a][OX2H1]", True),
    ("氨基", "[a][NX3H2]", True),
    ("硝基", "[a][N+](=O)[O-]", True),
    ("氰基", "[a][C]#N", True),
    ("醛基", "[a][CH1]=O", True),
    ("乙酰基", "[a][CX3](=O)[CH3]", True),
    ("酰基", "[#6;R]=[OX1]", True),
    ("甲基", "[a][CH3]", True),
    ("乙基", "[a][CH2][CH3]", True),
    ("氟", "[a]F", True),
    ("氯", "[a]Cl", True),
    ("羟甲基", "[a][CH2][OX2H1]", True),
]

_SUB_CACHE = [(nm, Chem.MolFromSmarts(sma)) for nm, sma, _ in SUB_PATTERNS]


def _ring_list(mol):
    return [list(r) for r in mol.GetRingInfo().AtomRings()]


def _choose_ring(mol):
    rings = _ring_list(mol)
    if not rings:
        return None
    arom6 = [r for r in rings if len(r) == 6 and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    if arom6:
        return sorted(arom6, key=lambda r: -sum(1 for i in r if mol.GetAtomWithIdx(i).GetSymbol() != "C"))[0]
    arom5 = [r for r in rings if len(r) == 5 and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    if arom5:
        return arom5[0]
    return max(rings, key=len)


def _parent_name(mol, ring):
    """返回 (母体名, 编号起点原子 idx 或 None, 是否苯环)。"""
    hetero = [(i, mol.GetAtomWithIdx(i).GetSymbol()) for i in ring
              if mol.GetAtomWithIdx(i).GetSymbol() != "C"]
    if len(ring) == 6 and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
        if not hetero:
            return "苯", None, True
        if len(hetero) == 1 and hetero[0][1] == "N":
            return "吡啶", hetero[0][0], False
        if len(hetero) == 2 and all(s == "N" for _, s in hetero):
            ps = sorted(ring.index(i) for i, _ in hetero)
            gap = min((ps[1] - ps[0]) % 6, (ps[0] - ps[1]) % 6)
            return {1: "哒嗪", 2: "嘧啶", 3: "吡嗪"}[gap], hetero[0][0], False
        return "", None, False
    if len(ring) == 5 and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
        syms = sorted(s for _, s in hetero)
        if syms == ["O"] or syms == ["S"] or syms == ["N"]:
            return {"O": "呋喃", "S": "噻吩", "N": "吡咯"}[syms[0]], hetero[0][0], False
        if syms == ["N", "N"]:
            ps = sorted(ring.index(i) for i, _ in hetero)
            gap = min((ps[1] - ps[0]) % 5, (ps[0] - ps[1]) % 5)
            return ("咪唑" if gap == 1 else "吡唑"), hetero[0][0], False
        return "", None, False
    if hetero:
        return "", None, False
    return {3: "环丙烷", 4: "环丁烷", 5: "环戊烷",
            6: "环己烷", 7: "环庚烷"}.get(len(ring), ""), None, False


def _substituents(mol, ring):
    """返回 {环上锚点原子: [取代基名...]}；出现认不出的取代基就返回 None。"""
    ring_set = set(ring)
    anchors = {}
    seen = set()
    for a in ring:
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            b = nb.GetIdx()
            if b in ring_set or b in seen:
                continue
            seen.add(b)
            name = None
            for nm, patt in _SUB_CACHE:
                ok = False
                for match in mol.GetSubstructMatches(patt):
                    if match[0] == a and match[1] == b:
                        ok = True
                        break
                if ok:
                    name = nm
                    break
            if name is None:
                # 认不出的取代基：宁可不出名字，也不瞎猜
                return None
            anchors.setdefault(a, []).append(name)
    return anchors


def _numbering(mol, ring, anchors, start, allowed):
    """给环编号，返回 {环原子: 位号}。合理的编号必须让杂原子落在规定位置。"""
    n = len(ring)
    hetero = {ring.index(i) for i in ring if mol.GetAtomWithIdx(i).GetSymbol() != "C"}
    best = None
    for s in range(n):
        if start is not None and ring[s] != start:
            continue
        for d in (1, -1):
            order = [ring[(s + d * k) % n] for k in range(n)]
            pos = {order[i]: i + 1 for i in range(n)}
            if allowed and not all(pos[ring[h]] in allowed for h in hetero):
                continue
            key = sorted(pos[a] for a in anchors)
            if best is None or key < best[0]:
                best = (key, pos)
        if start is not None and best is not None:
            break
    return best[1] if best else None


SUFFIX_OF = {"羟基": "酚", "氨基": "胺", "羧基": "甲酸", "醛基": "甲醛", "酰基": "酮"}
RING_KETONE = {"环丙烷": "环丙酮", "环丁烷": "环丁酮", "环戊烷": "环戊酮", "环己烷": "环己酮"}
RING_ALCOHOL = {"环丙烷": "环丙醇", "环丁烷": "环丁醇", "环戊烷": "环戊醇", "环己烷": "环己醇"}


def sysname(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    if any(a.GetFormalCharge() != 0 for a in mol.GetAtoms()):
        return ""
    if any(a.GetSymbol() not in "CNOFClS" for a in mol.GetAtoms()):
        return ""

    if len(_ring_list(mol)) > 1:          # 稠环/多环不猜名字
        return ""

    ring = _choose_ring(mol)
    if ring is None:
        return _chain_name(mol)

    parent, start, is_benzene = _parent_name(mol, ring)
    if not parent:
        return ""
    anchors = _substituents(mol, ring)
    if anchors is None:
        return ""
    if not anchors:
        return parent
    if sum(len(v) for v in anchors.values()) > 2:   # 取代基多于 2 个不猜
        return ""

    suffix_atom = None
    suffix = ""
    for a, names in anchors.items():
        if len(names) == 1 and names[0] in SUFFIX_OF:
            if names[0] in ("羟基", "氨基") and not (is_benzene or parent.startswith("环")):
                continue
            suffix_atom, suffix = a, SUFFIX_OF[names[0]]

    allowed = None
    if parent in ("吡啶", "吡咯", "呋喃", "噻吩"):
        allowed = {1}
    elif parent == "嘧啶":
        allowed = {1, 3}
    elif parent == "吡嗪":
        allowed = {1, 4}
    elif parent == "哒嗪":
        allowed = {1, 2}
    elif parent == "咪唑":
        allowed = {1, 3}
    elif parent == "吡唑":
        allowed = {1, 2}

    pos = _numbering(mol, ring, list(anchors), suffix_atom or start, allowed)
    if pos is None:
        return ""

    # 环酮 / 环醇 的特例
    if suffix == "酮" and parent in RING_KETONE:
        return RING_KETONE[parent]
    if suffix == "酚" and parent in RING_ALCOHOL:
        return RING_ALCOHOL[parent]

    parts = {}
    for a, names in anchors.items():
        for nm in names:
            if nm in SUFFIX_OF and suffix and SUFFIX_OF[nm] == suffix:
                continue
            parts.setdefault(nm, []).append(pos[a])

    total = sum(len(v) for v in parts.values())
    segs = []
    for nm, ps in parts.items():
        ps.sort()
        if is_benzene and not suffix and total == 1:
            segs.append(nm)
        elif len(ps) == 1:
            segs.append(f"{ps[0]}-{nm}")
        else:
            segs.append(f"{'、'.join(str(p) for p in ps)}-二{nm}")
    segs.sort(key=lambda s: (len(s), s))
    body = "".join(segs)
    if suffix == "甲酸":
        return body + "苯甲酸" if is_benzene else body + "酸"
    if suffix == "甲醛":
        return body + "苯甲醛" if is_benzene else body + "醛"
    if suffix:
        return body + parent + suffix
    return body + parent

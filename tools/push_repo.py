# -*- coding: utf-8 -*-
"""确认远端仓库是否存在，然后推送。token 只从命令行传入，不落盘。"""
import json
import sys
import urllib.error
import urllib.request

from dulwich import porcelain
from dulwich.repo import Repo

USER = "Toska2277"
REPO = "MolLoop"
TOKEN = sys.argv[1]
ROOT = r"C:\Users\den\Desktop\MolLoop"


def api(path, method="GET", payload=None):
    req = urllib.request.Request(
        "https://api.github.com" + path, method=method,
        data=json.dumps(payload).encode() if payload else None)
    req.add_header("Authorization", "Bearer " + TOKEN)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "MolLoop")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


try:
    me = api("/user")
    print("身份确认:", me["login"])
except urllib.error.HTTPError as e:
    print("token 校验失败:", e.code, e.read().decode()[:150])
    sys.exit(1)

exists = True
try:
    api(f"/repos/{USER}/{REPO}")
    print("仓库已存在")
except urllib.error.HTTPError as e:
    if e.code == 404:
        exists = False
        print("仓库不存在，尝试创建…")
    else:
        print("查询失败:", e.code)

if not exists:
    try:
        r = api("/user/repos", "POST", {"name": REPO, "private": False,
                                       "description": "MolLoop · QM9 分子能隙预测科研智能体（第五届未来杯 赛道二）"})
        print("已创建:", r["html_url"])
    except urllib.error.HTTPError as e:
        print("创建失败:", e.code, e.read().decode()[:200])
        print("→ 需要你先在网页上手动建空仓库")
        sys.exit(2)

rp = Repo(ROOT)
print("本地提交:", rp.head().decode()[:12])
url = f"https://{USER}:{TOKEN}@github.com/{USER}/{REPO}.git"
print("推送中…")
porcelain.push(ROOT, url, "refs/heads/main:refs/heads/main")
print(f"完成 → https://github.com/{USER}/{REPO}")

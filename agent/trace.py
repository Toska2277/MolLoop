# -*- coding: utf-8 -*-
"""全流程留痕：每一步写入 logs/trace.jsonl，保证研究过程可追溯、可复现。"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


class Tracer:
    """一个 run 对应一个 trace 文件，每步一行 JSON。"""

    def __init__(self, run_id: str | None = None, stage: str = "agent"):
        self.run_id = run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.stage = stage
        self.path = LOG_DIR / f"trace_{self.run_id}.jsonl"
        self._write("run_start", {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        })

    def _write(self, step: str, payload: dict) -> None:
        rec = {
            "run_id": self.run_id,
            "stage": self.stage,
            "step": step,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed": round(time.time() - (self.t0 if hasattr(self, "t0") else time.time()), 2),
            "payload": payload,
        }
        if not hasattr(self, "t0"):
            self.t0 = time.time()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def log(self, step: str, **payload) -> None:
        self._write(step, payload)

    def finish(self, status: str = "ok", **payload) -> None:
        self._write("run_end", {"status": status, **payload})

    @staticmethod
    def snapshot_config(cfg_path: Path, tag: str) -> Path:
        """每轮改动前保存配置快照，保证可回滚。"""
        snap_dir = LOG_DIR / "config_snapshots"
        snap_dir.mkdir(exist_ok=True)
        dst = snap_dir / f"{tag}_{time.strftime('%H%M%S')}.json"
        dst.write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
        return dst

    @staticmethod
    def run_cmd(args: list[str]) -> tuple[int, str]:
        """在项目根目录执行子进程，返回 (退出码, 输出尾部)。"""
        py = sys.executable
        proc = subprocess.run([py, *args], cwd=str(ROOT), capture_output=True,
                             text=True, encoding="utf-8", errors="replace")
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out[-3000:]

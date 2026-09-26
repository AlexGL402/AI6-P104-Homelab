#!/usr/bin/env python3
"""Standalone systemd runner for ForgeMiner managed by AI6 Host Monitor."""

import json
import os
import shlex
import shutil
import sys
import time
from pathlib import Path

STATE_DIR = Path(os.environ.get("AI6_MONITOR_STATE_DIR", str(Path.home() / ".local/state/ai6-monitor")))
CFG_PATH = Path(os.environ.get("AI6_MINER_CONFIG", str(STATE_DIR / "miner-config.json")))

DEFAULTS = {
    "algorithm": "pearlhash",
    "pool": "prl.kryptex.network:7048",
    "wallet": "",
    "worker": "ai6",
    "gpu": "0",
    "binary": "",
    "extra_args": "",
    "temp_limit": 80,
    "temp_resume": 70,
}


def load_cfg():
    cfg = dict(DEFAULTS)
    if CFG_PATH.is_file():
        data = json.loads(CFG_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            cfg.update(data)
    return cfg


def detect_binary(cfg):
    explicit = str(cfg.get("binary") or "").strip()
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())

    which = shutil.which("forge")
    if which:
        candidates.append(Path(which))

    home = Path.home()
    candidates += [
        home / "pearl" / "ForgeMiner" / "1.8.0" / "forge",
        home / "ForgeMiner" / "forge",
        home / "forgeminer" / "forge",
        home / "forge" / "forge",
        home / "miner" / "forge",
        home / "miners" / "forge" / "forge",
        Path("/opt/ForgeMiner/forge"),
        Path("/opt/forge/forge"),
        Path("/usr/local/bin/forge"),
    ]

    seen = set()
    for p in candidates:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


def main():
    cfg = load_cfg()
    binary = detect_binary(cfg)
    if not binary:
        print("ERROR: ForgeMiner binary not found; set it in the Miner tab and save config.", flush=True)
        return 2

    wallet = str(cfg.get("wallet") or "").strip()
    pool = str(cfg.get("pool") or "").strip()
    if not wallet:
        print("ERROR: miner wallet is empty.", flush=True)
        return 2
    if not pool:
        print("ERROR: miner pool is empty.", flush=True)
        return 2

    argv = [
        binary,
        "--algorithm", str(cfg.get("algorithm") or "pearlhash"),
        "--wallet", wallet,
        "--pool", pool,
        "--worker", str(cfg.get("worker") or "ai6"),
    ]

    gpu = str(cfg.get("gpu") or "").strip()
    if gpu:
        argv += ["--gpu", gpu]

    argv += ["--temp-limit", str(cfg.get("temp_limit") or 80)]
    argv += ["--temp-resume", str(cfg.get("temp_resume") or 70)]

    extra = str(cfg.get("extra_args") or "").strip()
    if extra:
        argv += shlex.split(extra)

    os.chdir(str(Path(binary).parent))
    print(
        "\n=== AI6 systemd miner start "
        + time.strftime("%Y-%m-%d %H:%M:%S %z")
        + " ===",
        flush=True,
    )
    os.execvpe(binary, argv, os.environ.copy())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise

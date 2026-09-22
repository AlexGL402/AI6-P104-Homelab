#!/usr/bin/env python3
"""ForgeMiner / PearlHash tab for the AI6 Host Monitor."""

import json
import html
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path

import psutil
from pydantic import BaseModel, Field
from fastapi.responses import PlainTextResponse

import ai6_monitor_dynamic as dynamic

base = dynamic.base

_STATE_DIR = Path(os.environ.get("AI6_MONITOR_STATE_DIR", str(Path.home() / ".local/state/ai6-monitor")))
_MINER_CFG = Path(os.environ.get("AI6_MINER_CONFIG", str(_STATE_DIR / "miner-config.json")))
_MINER_LOG = Path(os.environ.get("AI6_MINER_LOG", str(_STATE_DIR / "forge-miner.log")))
_MINER_PID = Path(os.environ.get("AI6_MINER_PID", str(_STATE_DIR / "forge-miner.pid")))
_MARKET_CACHE = {"ts": 0.0, "data": {}}
_KRYPTEX_CACHE = {"ts": 0.0, "wallet": "", "data": {}}
_MARKET_HISTORY = Path(os.environ.get("AI6_MARKET_HISTORY", str(_STATE_DIR / "prl-market-history.json")))

_DEFAULT_CFG = {
    "name": "Pearl / PearlHash",
    "algorithm": "pearlhash",
    "pool": "prl.kryptex.network:7048",
    "wallet": "",
    "worker": "ai6-cmp40",
    "gpu": "0",
    "binary": "",
    "extra_args": "",
    "temp_limit": 80,
    "temp_resume": 70,
    "energy_kzt_kwh": 35.0,
    "pool_fee_pct": 2.0,
}


class MinerConfigCommand(BaseModel):
    name: str = Field(default="Pearl / PearlHash", max_length=80)
    algorithm: str = Field(default="pearlhash", max_length=40)
    pool: str = Field(default="prl.kryptex.network:7048", max_length=240)
    wallet: str = Field(default="", max_length=240)
    worker: str = Field(default="ai6-cmp40", max_length=120)
    gpu: str = Field(default="0", max_length=100)
    binary: str = Field(default="", max_length=500)
    extra_args: str = Field(default="", max_length=1000)
    temp_limit: int = Field(default=80, ge=40, le=110)
    temp_resume: int = Field(default=70, ge=30, le=100)
    energy_kzt_kwh: float = Field(default=35.0, ge=0, le=10000)
    pool_fee_pct: float = Field(default=2.0, ge=0, le=100)


class MinerActionCommand(BaseModel):
    action: str


def _load_cfg():
    cfg = dict(_DEFAULT_CFG)
    try:
        if _MINER_CFG.is_file():
            data = json.loads(_MINER_CFG.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                cfg.update({k: v for k, v in data.items() if k in cfg})
    except Exception:
        pass
    return cfg


def _save_cfg(cfg):
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    _MINER_CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def _detect_binary(cfg=None):
    cfg = cfg or _load_cfg()
    explicit = str(cfg.get("binary") or "").strip()
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    which = shutil.which("forge")
    if which:
        candidates.append(Path(which))
    home = Path.home()
    candidates += [
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
        try:
            if p.is_file() and os.access(p, os.X_OK):
                return str(p)
        except Exception:
            pass
    return explicit or None


def _miner_proc():
    # Prefer the PID we created.
    try:
        if _MINER_PID.is_file():
            pid = int(_MINER_PID.read_text().strip())
            p = psutil.Process(pid)
            if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                return p
    except Exception:
        pass

    # Fallback: detect a forge process started elsewhere.
    for p in psutil.process_iter(["pid", "cmdline", "name", "create_time"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or [])
            name = str(p.info.get("name") or "")
            if re.search(r"(^|/|\s)forge(?:\s|$)", cmd, re.I) or name.lower() == "forge":
                return p
        except Exception:
            continue
    return None


def _vllm_busy():
    for p in psutil.process_iter(["cmdline"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or []).lower()
            if "vllm" in cmd and " serve " in (" " + cmd + " "):
                return True
        except Exception:
            pass
    return False


def _tail_log(max_bytes=120000):
    try:
        if not _MINER_LOG.is_file():
            return ""
        with _MINER_LOG.open("rb") as f:
            size = f.seek(0, os.SEEK_END)
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _parse_miner_log(text):
    clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text or "")
    lines = clean.splitlines()

    rate = None
    rate_unit = None
    pool_rate = None
    pool_rate_unit = None
    accepted = 0
    stale = 0
    rejected = 0
    pool_latency_ms = None
    avg_1m = None
    avg_1h = None
    avg_24h = None

    # Forge TUI table rows look like:
    # | 0  CMP 50HX 10G  74.61 TH/s  78.99 TH/s  10 / 0 / 0 |
    # | 1  CMP 50HX 10G  73.89 TH/s 102.94 TH/s  13 / 0 / 0 |
    # | -  2 GPU         148.50 TH/s 181.93 TH/s  23 / 0 / 0 |
    # Prefer the newest aggregate "- N GPU" row. This keeps the dashboard
    # hashrate/pool rate aligned with the total power draw when mining on
    # multiple GPUs. Fall back to GPU0 during early startup or older miner output.
    total_row_re = re.compile(
        r"\|\s*-\s+\d+\s+GPU\s+([0-9]+(?:\.[0-9]+)?)\s*(TH/s|GH/s|MH/s|kH/s|H/s)"
        r"\s+(?:(?:([0-9]+(?:\.[0-9]+)?)\s*(TH/s|GH/s|MH/s|kH/s|H/s))|--)"
        r"\s+(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*\|",
        re.I,
    )
    gpu_row_re = re.compile(
        r"\|\s*0\s+.+?\s+([0-9]+(?:\.[0-9]+)?)\s*(TH/s|GH/s|MH/s|kH/s|H/s)"
        r"\s+([0-9]+(?:\.[0-9]+)?)\s*(TH/s|GH/s|MH/s|kH/s|H/s)"
        r"\s+(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*\|",
        re.I,
    )
    for line in reversed(lines):
        m = total_row_re.search(line)
        if m:
            rate = float(m.group(1))
            rate_unit = m.group(2)
            pool_rate = float(m.group(3)) if m.group(3) is not None else None
            pool_rate_unit = m.group(4) if m.group(4) is not None else None
            accepted = int(m.group(5))
            stale = int(m.group(6))
            rejected = int(m.group(7))
            break

    if rate is None:
        for line in reversed(lines):
            m = gpu_row_re.search(line)
            if m:
                rate = float(m.group(1))
                rate_unit = m.group(2)
                pool_rate = float(m.group(3))
                pool_rate_unit = m.group(4)
                accepted = int(m.group(5))
                stale = int(m.group(6))
                rejected = int(m.group(7))
                break

    # Session Stats gives stable averages and latency; use the newest values.
    for line in reversed(lines):
        if pool_latency_ms is None:
            m = re.search(r"\|\s*Latency\s+~?\s*(\d+)\s*ms", line, re.I)
            if m:
                pool_latency_ms = int(m.group(1))
        if avg_1m is None:
            m = re.search(r"\|\s*Avg\s+1\s+min\s+([0-9]+(?:\.[0-9]+)?)\s*(TH/s|GH/s|MH/s|kH/s|H/s)", line, re.I)
            if m:
                avg_1m = float(m.group(1))
        if avg_1h is None:
            m = re.search(r"\|\s*Avg\s+1\s+hr\s+([0-9]+(?:\.[0-9]+)?)\s*(TH/s|GH/s|MH/s|kH/s|H/s)", line, re.I)
            if m:
                avg_1h = float(m.group(1))
        if avg_24h is None:
            m = re.search(r"\|\s*Avg\s+24\s+hr\s+([0-9]+(?:\.[0-9]+)?)\s*(TH/s|GH/s|MH/s|kH/s|H/s)", line, re.I)
            if m:
                avg_24h = float(m.group(1))
        if all(v is not None for v in (pool_latency_ms, avg_1m, avg_1h, avg_24h)):
            break

    # Fallback for very early startup before the first TUI table is printed.
    if rate is None:
        for line in reversed(lines):
            matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*(TH/s|GH/s|MH/s|kH/s|H/s)", line, re.I)
            if matches:
                val, unit = matches[0]
                rate = float(val)
                rate_unit = unit
                break

    return {
        "hashrate": rate,
        "hashrate_unit": rate_unit,
        "pool_hashrate": pool_rate,
        "pool_hashrate_unit": pool_rate_unit,
        "avg_1m": avg_1m,
        "avg_1h": avg_1h,
        "avg_24h": avg_24h,
        "accepted": accepted,
        "stale": stale,
        "rejected": rejected,
        "latency_ms": pool_latency_ms,
    }


def _gpu_snapshot():
    out = []
    try:
        p = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,power.draw,power.limit,temperature.gpu,fan.speed,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=3.0,
        )
        if p.returncode != 0:
            return out
        for line in p.stdout.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 9:
                continue
            def num(v):
                try:
                    return float(v)
                except Exception:
                    return None
            out.append({
                "index": int(float(parts[0])),
                "name": parts[1],
                "load_pct": num(parts[2]),
                "power_w": num(parts[3]),
                "power_limit_w": num(parts[4]),
                "temp_c": num(parts[5]),
                "fan_pct": num(parts[6]),
                "vram_used_mib": num(parts[7]),
                "vram_total_mib": num(parts[8]),
            })
    except Exception:
        pass
    return out



def _http_get(url, timeout=5.0):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "AI6-Miner-Monitor/1.0",
            "Accept": "application/json,text/html,application/xml,text/xml,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace"), (r.headers.get("content-type") or "")


def _walk_json(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_json(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_json(v)


def _num(v):
    try:
        if isinstance(v, str):
            v = v.replace(",", "").strip()
        return float(v)
    except Exception:
        return None


def _scaled_number(value, suffix=""):
    v = _num(value)
    if v is None:
        return None
    mult = {
        "": 1.0, "k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12,
        "p": 1e15, "e": 1e18,
    }.get(str(suffix or "").strip().lower(), 1.0)
    return v * mult


def _extract_pearltrack_live():
    """Read the explorer's current headline stats (price + network) from one source."""
    try:
        raw, _ = _http_get("https://www.pearltrack.io/")
        text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
        text = re.sub(r"\s+", " ", text)

        out = {"source": "PearlTrack"}

        m = re.search(r"PRL Price\s*\$?\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
        if m:
            out["prl_usdt"] = _num(m.group(1))

        m = re.search(
            r"Network Hashrate\s*([0-9]+(?:\.[0-9]+)?)\s*([kKmMgGtTpPeE]?H/s)",
            text, re.I
        )
        if m:
            unit = m.group(2).lower()
            unit_mult = {
                "h/s": 1.0, "kh/s": 1e3, "mh/s": 1e6, "gh/s": 1e9,
                "th/s": 1e12, "ph/s": 1e15, "eh/s": 1e18,
            }.get(unit)
            if unit_mult:
                out["network_hashrate_hs"] = float(m.group(1)) * unit_mult

        m = re.search(r"Difficulty\s*([0-9]+(?:\.[0-9]+)?)\s*([kKmMbBtT]?)", text, re.I)
        if m:
            out["difficulty"] = _scaled_number(m.group(1), m.group(2))

        m = re.search(r"~\s*([0-9]+(?:\.[0-9]+)?)\s*s\s*/\s*block", text, re.I)
        if m:
            out["block_time_s"] = _num(m.group(1))

        m = re.search(r"Block Reward\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*PRL", text, re.I)
        if m:
            out["block_reward_prl"] = _num(m.group(1))

        m = re.search(r"Block Height\s*([0-9][0-9,]*)", text, re.I)
        if m:
            h = _num(m.group(1))
            out["height"] = int(h) if h else None

        if any(out.get(k) is not None for k in (
            "prl_usdt", "network_hashrate_hs", "difficulty",
            "block_time_s", "block_reward_prl"
        )):
            return out
    except Exception:
        pass
    return {}


def _extract_safetrade_price():
    # SafeTrade's public API has changed paths over time, so try known public
    # ticker routes first and then fall back to the public markets page.
    candidates = [
        "https://safe.trade/api/v2/public/markets/prlusdt/tickers",
        "https://safe.trade/api/v2/public/markets/tickers",
        "https://safetrade.com/api/v2/public/markets/prlusdt/tickers",
        "https://safetrade.com/api/v2/public/markets/tickers",
    ]
    for url in candidates:
        try:
            raw, _ = _http_get(url)
            data = json.loads(raw)
            for node in _walk_json(data):
                market = str(
                    node.get("market")
                    or node.get("symbol")
                    or node.get("id")
                    or node.get("market_id")
                    or ""
                ).lower().replace("_", "").replace("-", "").replace("/", "")
                if market and market != "prlusdt":
                    continue
                for key in ("last", "last_price", "lastPrice", "close", "price"):
                    p = _num(node.get(key))
                    if p and p > 0:
                        return p, "SafeTrade API"
        except Exception:
            pass

    try:
        raw, _ = _http_get("https://safetrade.com/markets")
        # The public page renders a PRL/USDT row. Keep the expression loose
        # enough to survive minor markup changes.
        text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
        text = re.sub(r"\s+", " ", text)
        m = re.search(
            r"PRL\s*/\s*USDT\s*(?:[+-]?[0-9]+(?:\.[0-9]+)?%\s*)?([0-9]+(?:\.[0-9]+)?)",
            text,
            re.I,
        )
        if m:
            p = _num(m.group(1))
            if p and p > 0:
                return p, "SafeTrade markets"
    except Exception:
        pass
    return None, None


def _extract_usd_kzt():
    # Official National Bank of Kazakhstan feed first.
    try:
        raw, _ = _http_get("https://nationalbank.kz/rss/rates_all.xml")
        root = ET.fromstring(raw)
        for item in root.iter():
            children = list(item)
            vals = {str(ch.tag).split("}")[-1].lower(): (ch.text or "").strip() for ch in children}
            if vals.get("title", "").upper() == "USD":
                v = _num(vals.get("description") or vals.get("rate"))
                if v and v > 0:
                    return v, "NBK"
    except Exception:
        pass

    # Public no-key fallback.
    try:
        raw, _ = _http_get("https://open.er-api.com/v6/latest/USD")
        data = json.loads(raw)
        v = _num((data.get("rates") or {}).get("KZT"))
        if v and v > 0:
            return v, "ExchangeRate API"
    except Exception:
        pass
    return None, None


def _extract_network():
    out = {
        "network_hashrate_hs": None,
        "difficulty": None,
        "block_reward_prl": None,
        "block_time_s": None,
        "height": None,
        "source": None,
    }

    # PearlNet exposes pool-wide live network stats without auth.
    try:
        raw, _ = _http_get("https://pearl-net.com/api/pool/stats")
        data = json.loads(raw)
        for node in _walk_json(data):
            for key in ("networkHashrate", "network_hashrate", "networkHashRate", "netHashrate"):
                v = _num(node.get(key))
                if v and v > 0:
                    out["network_hashrate_hs"] = v
            for key in ("difficulty", "networkDifficulty", "network_difficulty"):
                v = _num(node.get(key))
                if v and v > 0:
                    out["difficulty"] = v
            for key in ("blockReward", "block_reward", "reward"):
                v = _num(node.get(key))
                if v and v > 0 and v < 1e9:
                    out["block_reward_prl"] = v
            for key in ("height", "blockHeight", "block_height"):
                v = _num(node.get(key))
                if v and v > 0:
                    out["height"] = int(v)
        if any(out[k] is not None for k in ("network_hashrate_hs", "block_reward_prl", "difficulty")):
            out["source"] = "PearlNet"
    except Exception:
        pass

    # PearlTrack latest blocks: fill difficulty/reward and estimate recent block time.
    block_urls = [
        "https://www.pearltrack.io/api/v1/blocks?page=1&limit=20",
        "https://www.pearltrack.io/api/v1/blocks?limit=20",
    ]
    for url in block_urls:
        try:
            raw, _ = _http_get(url)
            data = json.loads(raw)
            blocks = None
            if isinstance(data, list):
                blocks = data
            elif isinstance(data, dict):
                for key in ("blocks", "items", "data", "results"):
                    if isinstance(data.get(key), list):
                        blocks = data[key]
                        break
            if not blocks:
                continue

            b0 = blocks[0] if isinstance(blocks[0], dict) else {}
            if out["difficulty"] is None:
                out["difficulty"] = _num(b0.get("difficulty"))
            if out["block_reward_prl"] is None:
                out["block_reward_prl"] = (
                    _num(b0.get("rewardPrl"))
                    or _num(b0.get("blockRewardPrl"))
                    or _num(b0.get("reward"))
                )
            if out["height"] is None:
                h = _num(b0.get("height") or b0.get("blockHeight"))
                out["height"] = int(h) if h else None

            times = []
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                t = _num(b.get("timeMs") or b.get("timestampMs") or b.get("timestamp") or b.get("time"))
                if t:
                    if t > 1e12:
                        t /= 1000.0
                    times.append(t)
            times = sorted(times, reverse=True)
            diffs = [times[i] - times[i+1] for i in range(len(times)-1) if 5 <= times[i] - times[i+1] <= 7200]
            if diffs:
                diffs.sort()
                out["block_time_s"] = diffs[len(diffs)//2]
            out["source"] = "PearlTrack" if out["source"] is None else out["source"] + " + PearlTrack"
            break
        except Exception:
            pass

    return out


def _live_market_data():
    now = time.time()
    if now - float(_MARKET_CACHE.get("ts") or 0) < 60 and _MARKET_CACHE.get("data"):
        return _MARKET_CACHE["data"]

    # PearlTrack is the primary source because its homepage exposes mutually
    # consistent price, network hashrate, difficulty, block time and reward.
    # It also tracks the SafeTrade PRL market. Fallbacks fill only missing fields.
    pt = _extract_pearltrack_live()
    usd_kzt, fx_source = _extract_usd_kzt()

    price = pt.get("prl_usdt")
    price_source = "PearlTrack (SafeTrade)" if price else None
    if price is None:
        price, price_source = _extract_safetrade_price()

    network = {
        "network_hashrate_hs": pt.get("network_hashrate_hs"),
        "difficulty": pt.get("difficulty"),
        "block_reward_prl": pt.get("block_reward_prl"),
        "block_time_s": pt.get("block_time_s"),
        "height": pt.get("height"),
        "source": pt.get("source"),
    }

    # Only use legacy network extraction if PearlTrack failed entirely.
    if not all(network.get(k) is not None for k in (
        "network_hashrate_hs", "difficulty", "block_reward_prl", "block_time_s"
    )):
        fallback = _extract_network()
        for k, v in fallback.items():
            if network.get(k) is None and v is not None:
                network[k] = v
        if network.get("source") and fallback.get("source"):
            network["source"] += " + " + fallback["source"]
        elif fallback.get("source"):
            network["source"] = fallback["source"]

    data = {
        "prl_usdt": price,
        "price_source": price_source,
        "usd_kzt": usd_kzt,
        "fx_source": fx_source,
        **network,
        "updated_at": int(now),
    }
    _MARKET_CACHE["ts"] = now
    _MARKET_CACHE["data"] = data
    return data

def _to_hs(rate, unit):
    if rate is None:
        return None
    mult = {
        "h/s": 1.0,
        "kh/s": 1e3,
        "mh/s": 1e6,
        "gh/s": 1e9,
        "th/s": 1e12,
        "ph/s": 1e15,
        "eh/s": 1e18,
    }.get(str(unit or "").lower())
    return float(rate) * mult if mult else None


def _profitability(parsed, gpus, cfg):
    live = _live_market_data()
    miner_hs = _to_hs(parsed.get("avg_1m") or parsed.get("hashrate"), parsed.get("hashrate_unit") or "TH/s")
    power_w = sum(float(g.get("power_w") or 0) for g in gpus)
    eff = None
    display_hashrate = parsed.get("avg_1m") or parsed.get("hashrate")
    if power_w > 0 and display_hashrate is not None:
        # Use the same aggregate miner rate as profitability and divide by
        # aggregate GPU power. This avoids showing half efficiency on 2+ GPUs.
        eff = float(display_hashrate) / power_w

    prl_day = None
    net_hs = live.get("network_hashrate_hs")
    reward = live.get("block_reward_prl")
    block_time = live.get("block_time_s")
    if miner_hs and net_hs and reward and block_time and net_hs > 0 and block_time > 0:
        share = miner_hs / net_hs
        blocks_day = 86400.0 / block_time
        fee = max(0.0, min(100.0, float(cfg.get("pool_fee_pct") or 0))) / 100.0
        prl_day = share * blocks_day * reward * (1.0 - fee)

    price = live.get("prl_usdt")
    usd_kzt = live.get("usd_kzt")
    gross_day_usdt = prl_day * price if prl_day is not None and price else None
    gross_day_kzt = gross_day_usdt * usd_kzt if gross_day_usdt is not None and usd_kzt else None

    energy_rate = float(cfg.get("energy_kzt_kwh") or 0)
    energy_day_kzt = (power_w / 1000.0) * 24.0 * energy_rate if power_w > 0 else 0.0
    net_day_kzt = gross_day_kzt - energy_day_kzt if gross_day_kzt is not None else None

    def scale(v, factor):
        return None if v is None else v * factor

    return {
        "efficiency_hashrate_per_w": eff,
        "miner_hashrate": parsed.get("avg_1m") or parsed.get("hashrate"),
        "miner_hashrate_unit": parsed.get("hashrate_unit") or "TH/s",
        "power_w": power_w,
        "energy_kzt_kwh": energy_rate,
        "pool_fee_pct": float(cfg.get("pool_fee_pct") or 0),
        "prl_hour": scale(prl_day, 1/24),
        "prl_day": prl_day,
        "prl_month": scale(prl_day, 30),
        "gross_usdt_hour": scale(gross_day_usdt, 1/24),
        "gross_usdt_day": gross_day_usdt,
        "gross_usdt_month": scale(gross_day_usdt, 30),
        "gross_kzt_hour": scale(gross_day_kzt, 1/24),
        "gross_kzt_day": gross_day_kzt,
        "gross_kzt_month": scale(gross_day_kzt, 30),
        "electricity_kzt_hour": scale(energy_day_kzt, 1/24),
        "electricity_kzt_day": energy_day_kzt,
        "electricity_kzt_month": scale(energy_day_kzt, 30),
        "net_kzt_hour": scale(net_day_kzt, 1/24),
        "net_kzt_day": net_day_kzt,
        "net_kzt_month": scale(net_day_kzt, 30),
        "market": live,
    }



def _kryptex_wallet_stats(wallet):
    wallet = str(wallet or "").strip()
    if not wallet:
        return {}

    now = time.time()
    if (
        _KRYPTEX_CACHE.get("wallet") == wallet
        and now - float(_KRYPTEX_CACHE.get("ts") or 0) < 60
        and _KRYPTEX_CACHE.get("data")
    ):
        return _KRYPTEX_CACHE["data"]

    base_url = f"https://pool.kryptex.com/prl/miner/stats/{wallet}"
    out = {
        "url": base_url,
        "payouts_url": f"https://pool.kryptex.com/prl/miner/payouts/{wallet}",
        "settings_url": f"https://pool.kryptex.com/prl/miner/settings/{wallet}",
        "workers_online": None,
        "workers_offline": None,
        "hashrate_30m_ths": None,
        "hashrate_3h_ths": None,
        "hashrate_24h_ths": None,
        "confirmed_prl": None,
        "confirmed_usd": None,
        "pending_prl": None,
        "pending_usd": None,
        "reward_7d_prl": None,
        "reward_30d_prl": None,
        "total_paid_prl": None,
        "worker_name": None,
        "worker_mode": None,
        "worker_valid": None,
        "worker_stale": None,
        "worker_invalid": None,
        "worker_miner": None,
        "updated_at": int(now),
    }

    try:
        raw, _ = _http_get(base_url, timeout=8.0)
        text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
        text = re.sub(r"\s+", " ", text)

        def grab(pattern, conv=float):
            m = re.search(pattern, text, re.I)
            if not m:
                return None
            try:
                return conv(m.group(1).replace(",", ""))
            except Exception:
                return None

        # Summary block.
        out["workers_online"] = grab(r"Workers\s+(\d+)\s+Online", int)
        out["workers_offline"] = grab(r"Online\s+\d+\s+Offline\s+(\d+)", int)
        out["hashrate_30m_ths"] = grab(r"Hashrate\s+([0-9]+(?:\.[0-9]+)?)\s*TH/s\s+30\s*min")
        out["hashrate_3h_ths"] = grab(r"30\s*min\s+([0-9]+(?:\.[0-9]+)?)\s*TH/s\s+3\s*h")
        out["hashrate_24h_ths"] = grab(r"3\s*h\s+([0-9]+(?:\.[0-9]+)?)\s*TH/s\s+24\s*h")

        # Balance and rewards.
        out["confirmed_prl"] = grab(r"Confirmed\s+([0-9]+(?:\.[0-9]+)?)\s*PRL")
        out["confirmed_usd"] = grab(r"Confirmed.*?PRL\s*[≈~]?\s*\$?([0-9]+(?:\.[0-9]+)?)\s*USD")
        out["pending_prl"] = grab(r"Pending\s+([0-9]+(?:\.[0-9]+)?)\s*PRL")
        out["pending_usd"] = grab(r"Pending.*?PRL\s*[≈~]?\s*\$?([0-9]+(?:\.[0-9]+)?)\s*USD")
        out["reward_7d_prl"] = grab(r"Reward\s*\(\s*7\s*days\s*\)\s+([0-9]+(?:\.[0-9]+)?)\s*PRL")
        out["reward_30d_prl"] = grab(r"Reward\s*\(\s*30\s*days\s*\)\s+([0-9]+(?:\.[0-9]+)?)\s*PRL")
        out["total_paid_prl"] = grab(r"Total\s+Paid\s+([0-9]+(?:\.[0-9]+)?)\s*PRL")

        # Worker row. Use configured worker name if visible, otherwise take the
        # first PPS+/SOLO row that follows the workers table.
        worker = str((_load_cfg().get("worker") or "")).strip()
        row = None
        if worker:
            row = re.search(
                re.escape(worker)
                + r"\s+(PPS\+|SOLO)\s+([0-9]+(?:\.[0-9]+)?)\s*TH/s\s+"
                  r"([0-9]+(?:\.[0-9]+)?)\s*TH/s\s+(\d+)\s+(\d+)\s+(\d+)",
                text,
                re.I,
            )
        if row:
            out["worker_name"] = worker
            out["worker_mode"] = row.group(1).upper()
            out["worker_valid"] = int(row.group(4))
            out["worker_stale"] = int(row.group(5))
            out["worker_invalid"] = int(row.group(6))

            # Miner software often appears just after the worker row.
            pos = row.end()
            m = re.search(r"(ForgeMiner/[0-9.]+|lolMiner/[0-9.]+|Rigel/[0-9.]+|SRBMiner[^ ]*)", text[pos:pos+500], re.I)
            if m:
                out["worker_miner"] = m.group(1)

    except Exception as e:
        out["error"] = str(e)

    _KRYPTEX_CACHE["ts"] = now
    _KRYPTEX_CACHE["wallet"] = wallet
    _KRYPTEX_CACHE["data"] = out
    return out



def _record_market_history(profitability):
    """Persist a compact PRL/network/profit snapshot at most every 10 minutes."""
    try:
        p = profitability or {}
        m = p.get("market") or {}
        price = m.get("prl_usdt")
        net = p.get("net_kzt_day")
        if price is None and net is None:
            return

        now = int(time.time())
        rows = []
        if _MARKET_HISTORY.is_file():
            try:
                data = json.loads(_MARKET_HISTORY.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    rows = [x for x in data if isinstance(x, dict)]
            except Exception:
                rows = []

        if rows and now - int(rows[-1].get("ts") or 0) < 600:
            return

        row = {
            "ts": now,
            "price_usdt": price,
            "network_hashrate_hs": m.get("network_hashrate_hs"),
            "difficulty": m.get("difficulty"),
            "block_time_s": m.get("block_time_s"),
            "block_reward_prl": m.get("block_reward_prl"),
            "hashrate": p.get("miner_hashrate"),
            "hashrate_unit": p.get("miner_hashrate_unit"),
            "power_w": p.get("power_w"),
            "gross_kzt_day": p.get("gross_kzt_day"),
            "electricity_kzt_day": p.get("electricity_kzt_day"),
            "net_kzt_day": net,
            "prl_day": p.get("prl_day"),
            "usd_kzt": m.get("usd_kzt"),
        }
        rows.append(row)

        # Keep up to 45 days at 10-minute cadence with some margin.
        cutoff = now - 45 * 86400
        rows = [x for x in rows if int(x.get("ts") or 0) >= cutoff][-7000:]
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _MARKET_HISTORY.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        tmp.replace(_MARKET_HISTORY)
    except Exception:
        pass


def _status():
    cfg = _load_cfg()
    proc = _miner_proc()
    running = proc is not None
    uptime = None
    pid = None
    if proc:
        try:
            pid = proc.pid
            uptime = max(0, time.time() - proc.create_time())
        except Exception:
            pass
    log = _tail_log()
    parsed = _parse_miner_log(log)
    binary = _detect_binary(cfg)
    gpus = _gpu_snapshot()
    profitability = _profitability(parsed, gpus, cfg)
    _record_market_history(profitability)
    kryptex = _kryptex_wallet_stats(cfg.get("wallet"))
    return {
        "running": running,
        "pid": pid,
        "uptime_s": uptime,
        "vllm_busy": _vllm_busy(),
        "binary_detected": binary,
        "config": cfg,
        "metrics": parsed,
        "gpus": gpus,
        "profitability": profitability,
        "kryptex": kryptex,
        "log_path": str(_MINER_LOG),
    }


@base.app.get("/api/miner/status")
def api_miner_status():
    return _status()


@base.app.post("/api/miner/config")
def api_miner_config(cmd: MinerConfigCommand):
    cfg = cmd.model_dump()
    _save_cfg(cfg)
    return {"ok": True, "config": cfg, "binary_detected": _detect_binary(cfg)}


@base.app.post("/api/miner/control")
def api_miner_control(cmd: MinerActionCommand):
    action = (cmd.action or "").strip().lower()

    if action == "start":
        if _miner_proc():
            return {"ok": True, "message": "Miner is already running."}
        if _vllm_busy():
            raise base.HTTPException(status_code=409, detail="vLLM is running. Stop AI inference before starting the miner.")

        cfg = _load_cfg()
        binary = _detect_binary(cfg)
        if not binary:
            raise base.HTTPException(
                status_code=400,
                detail="ForgeMiner binary not found. Set the Miner binary field (path to forge) and Save config.",
            )
        if not str(cfg.get("wallet") or "").strip():
            raise base.HTTPException(status_code=400, detail="Wallet is empty. Enter the PRL wallet and Save config.")
        if not str(cfg.get("pool") or "").strip():
            raise base.HTTPException(status_code=400, detail="Pool is empty.")

        argv = [
            binary,
            "--algorithm", str(cfg.get("algorithm") or "pearlhash"),
            "--wallet", str(cfg["wallet"]),
            "--pool", str(cfg["pool"]),
            "--worker", str(cfg.get("worker") or "ai6"),
        ]
        gpu = str(cfg.get("gpu") or "").strip()
        if gpu:
            argv += ["--gpu", gpu]
        argv += ["--temp-limit", str(cfg.get("temp_limit") or 80)]
        argv += ["--temp-resume", str(cfg.get("temp_resume") or 70)]
        extra = str(cfg.get("extra_args") or "").strip()
        if extra:
            try:
                argv += shlex.split(extra)
            except ValueError as e:
                raise base.HTTPException(status_code=400, detail=f"Invalid extra args: {e}")

        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        logf = _MINER_LOG.open("ab", buffering=0)
        header = ("\n\n=== AI6 monitor start " + time.strftime("%Y-%m-%d %H:%M:%S %z") + " ===\n").encode()
        logf.write(header)
        proc = subprocess.Popen(
            argv,
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=str(Path(binary).parent),
            start_new_session=True,
            env=os.environ.copy(),
        )
        logf.close()
        _MINER_PID.write_text(str(proc.pid), encoding="utf-8")
        return {"ok": True, "message": "Miner started.", "pid": proc.pid}

    if action == "stop":
        proc = _miner_proc()
        if not proc:
            try:
                _MINER_PID.unlink(missing_ok=True)
            except Exception:
                pass
            return {"ok": True, "message": "Miner is not running."}
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        try:
            _MINER_PID.unlink(missing_ok=True)
        except Exception:
            pass
        return {"ok": True, "message": "Miner stopped."}

    raise base.HTTPException(status_code=400, detail="action must be start or stop")


@base.app.get("/api/miner/logs")
def api_miner_logs():
    return PlainTextResponse(_tail_log(), media_type="text/plain; charset=utf-8")


def install():
    dashboard = base.DASHBOARD

    # Add a top-level Miner tab next to Benchs.
    dashboard = dashboard.replace(
        '<button id="tabBenchsBtn" class="tab-btn" onclick="showTopTab(\'benchs\')">Benchs</button>',
        '<button id="tabBenchsBtn" class="tab-btn" onclick="showTopTab(\'benchs\')">Benchs</button>\n'
        '<button id="tabMinerBtn" class="tab-btn" onclick="showTopTab(\'miner\')">Miner</button>',
        1,
    )

    miner_html = r'''
<div id="minerTab" style="display:none">
  <div class="section">
    <div class="section-head">
      <div>
        <div class="section-title">Pearl / ForgeMiner</div>
        <div class="section-sub">Mine on idle GPUs. Start is blocked while vLLM is running.</div>
      </div>
      <div id="minerState" class="status down">● STOPPED</div>
    </div>

    <div class="miner-summary">
      <div class="miner-kpi"><span>Hashrate</span><b id="minerHashrate">—</b><small id="minerAlgo">PearlHash</small></div>
      <div class="miner-kpi"><span>Shares</span><b id="minerShares">—</b><small id="minerLatency">accepted / rejected</small></div>
      <div class="miner-kpi"><span>Uptime</span><b id="minerUptime">—</b><small id="minerPid">PID —</small></div>
      <div class="miner-kpi"><span>AI status</span><b id="minerAiState">—</b><small>miner start interlock</small></div>
    </div>

    <div class="miner-config">
      <label>Name<input id="minerName" value="Pearl / PearlHash"></label>
      <label>Pool<input id="minerPool" value="prl.kryptex.network:7048"></label>
      <label>Wallet<input id="minerWallet" placeholder="PRL wallet"></label>
      <label>Worker<input id="minerWorker" value="ai6-cmp40"></label>
      <label>Algorithm<input id="minerAlgorithm" value="pearlhash"></label>
      <label>GPU(s)<input id="minerGpu" value="0" placeholder="0 or 0,1"></label>
      <label>Miner binary<input id="minerBinary" placeholder="/path/to/forge"></label>
      <label>Extra args<input id="minerExtra" placeholder="e.g. --cmp-install"></label>
      <label>Temp limit °C<input id="minerTempLimit" type="number" value="80" min="40" max="110"></label>
      <label>Resume °C<input id="minerTempResume" type="number" value="70" min="30" max="100"></label>
      <label>Electricity ₸/kWh<input id="minerEnergyKzt" type="number" value="35" min="0" step="0.1"></label>
      <label>Pool fee %<input id="minerPoolFee" type="number" value="2" min="0" max="100" step="0.1"></label>
    </div>

    <div class="miner-actions">
      <button type="button" onclick="saveMinerConfig()">Save config</button>
      <button type="button" class="miner-start" onclick="controlMiner('start')">Start</button>
      <button type="button" class="miner-stop" onclick="controlMiner('stop')">Stop</button>
      <span id="minerMsg" class="muted"></span>
    </div>

    <div class="miner-power-row">
      <label>GPU power limit
        <select id="minerPlPreset">
          <option value="125">125 W</option>
          <option value="130" selected>130 W</option>
          <option value="140">140 W</option>
          <option value="150">150 W</option>
          <option value="165">165 W</option>
          <option value="184">184 W</option>
          <option value="200">200 W</option>
          <option value="220">220 W</option>
        </select>
      </label>
      <label>GPU index<input id="minerPlGpu" type="number" value="0" min="0" max="31"></label>
      <button type="button" onclick="setMinerPowerLimit()">Set PL</button>
      <span id="minerPlMsg" class="muted"></span>
    </div>

    <div id="minerGpuGrid" class="vllm-gpu-grid"></div>

    <div class="miner-profit">
      <div class="miner-kpi"><span>Efficiency</span><b id="minerEfficiency">—</b><small id="minerEfficiencySub">GPU hashrate / GPU watts</small></div>
      <div class="miner-kpi"><span>PRL price</span><b id="minerPrlPrice">—</b><small id="minerPriceSource">live market</small></div>
      <div class="miner-kpi"><span>Network</span><b id="minerNetwork">—</b><small id="minerDifficulty">difficulty —</small></div>
      <div class="miner-kpi"><span>Electricity</span><b id="minerElectricity">—</b><small id="minerEnergyRate">35 ₸/kWh • GPU only</small></div>
    </div>

    <div class="miner-profit-table">
      <div class="mph"><span>Period</span><b>PRL</b><b>Gross ₸</b><b>Electricity ₸</b><b>Net ₸</b></div>
      <div class="mpr"><span>Hour</span><b id="profitPrlHour">—</b><b id="profitGrossHour">—</b><b id="profitElecHour">—</b><b id="profitNetHour">—</b></div>
      <div class="mpr"><span>Day</span><b id="profitPrlDay">—</b><b id="profitGrossDay">—</b><b id="profitElecDay">—</b><b id="profitNetDay">—</b></div>
      <div class="mpr"><span>30 days</span><b id="profitPrlMonth">—</b><b id="profitGrossMonth">—</b><b id="profitElecMonth">—</b><b id="profitNetMonth">—</b></div>
    </div>

    <div class="miner-profit-note" id="minerProfitNote">Profitability uses live PRL price/network data and current GPU power. Mining income is an estimate and varies with network difficulty, pool luck and price.</div>

    <div class="miner-kryptex-panel">
      <div class="miner-kryptex-head">
        <div><b>Kryptex Pool stats</b><small id="kryptexUpdated">public miner page • refresh ~60s</small></div>
        <div class="miner-kryptex-links">
          <a id="kryptexStatsLink" href="#" target="_blank" rel="noopener">Workers ↗</a>
          <a id="kryptexPayoutsLink" href="#" target="_blank" rel="noopener">Payouts ↗</a>
          <a id="kryptexSettingsLink" href="#" target="_blank" rel="noopener">Settings ↗</a>
        </div>
      </div>
      <div class="miner-kryptex-grid">
        <div><span>Workers</span><b id="kxWorkers">—</b><small>online / offline</small></div>
        <div><span>Pool hashrate</span><b id="kxHash30m">—</b><small id="kxHashLong">3h — • 24h —</small></div>
        <div><span>Balance</span><b id="kxBalance">—</b><small id="kxPending">pending —</small></div>
        <div><span>Payouts</span><b id="kxPaid">—</b><small id="kxRewards">7d — • 30d —</small></div>
        <div><span>Worker shares</span><b id="kxShares">—</b><small>valid / stale / invalid</small></div>
        <div><span>Worker</span><b id="kxWorker">—</b><small id="kxMiner">—</small></div>
      </div>
      <div id="kxError" class="miner-profit-note"></div>
    </div>

    <div class="miner-log-wrap">
      <div class="miner-log-head"><span id="minerLogPath">ForgeMiner log</span><button type="button" onclick="refreshMinerLogs()">Refresh log</button></div>
      <pre id="minerLogText">No miner log loaded yet.</pre>
    </div>
  </div>
</div>
'''
    if "</body>" not in dashboard:
        raise RuntimeError("Miner tab injection failed: </body> marker not found")
    if 'id="minerTab"' not in dashboard:
        dashboard = dashboard.replace("</body>", miner_html + "\n</body>", 1)

    css = r'''
.miner-summary,.miner-profit{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:8px;margin:10px 0}
.miner-profit-table{border:1px solid #30343a;border-radius:9px;overflow:hidden;margin:9px 0}.miner-profit-table .mph,.miner-profit-table .mpr{display:grid;grid-template-columns:120px repeat(4,minmax(100px,1fr));gap:8px;padding:7px 10px;align-items:center}.miner-profit-table .mph{background:#121416;color:#8f969c;font-size:10px}.miner-profit-table .mpr{border-top:1px solid #292d31;font-size:11px}.miner-profit-table .mpr b:last-child{color:#7be495}.miner-profit-note{color:#8d949a;font-size:10px;margin:4px 0 10px}
.miner-kpi{background:#171717;border:1px solid #303030;border-radius:9px;padding:9px 11px}
.miner-kpi span,.miner-kpi small{display:block;color:#92979c;font-size:10px}.miner-kpi b{display:block;color:#eee;font-size:21px;margin:2px 0}
.miner-config{display:grid;grid-template-columns:repeat(5,minmax(150px,1fr));gap:8px;align-items:end}
.miner-config label,.miner-power-row label{font-size:10px;color:#aaa}.miner-config input,.miner-power-row input,.miner-power-row select{display:block;width:100%;margin-top:4px;padding:7px;background:#111;color:#ddd;border:1px solid #3a3a3a;border-radius:6px}
.miner-actions,.miner-power-row{display:flex;gap:7px;align-items:end;flex-wrap:wrap;margin-top:10px}.miner-power-row label{min-width:120px}.miner-power-row button{height:32px}
.miner-start{border-color:#2f7540!important;color:#72e28a!important;background:#132519!important}.miner-stop{border-color:#7a3434!important;color:#ff8585!important;background:#2a1515!important}
.miner-kryptex-panel{margin-top:10px;background:#121416;border:1px solid #30343a;border-radius:9px;padding:10px}.miner-kryptex-head{display:flex;justify-content:space-between;gap:10px;align-items:center;flex-wrap:wrap}.miner-kryptex-head b{display:block}.miner-kryptex-head small{display:block;color:#8f969c;font-size:10px;margin-top:2px}.miner-kryptex-links{display:flex;gap:6px;flex-wrap:wrap}.miner-kryptex-links a{color:#79bfff;text-decoration:none;border:1px solid #315d7b;background:#14212a;border-radius:6px;padding:5px 8px;font-size:10px}.miner-kryptex-grid{display:grid;grid-template-columns:repeat(6,minmax(140px,1fr));gap:7px;margin-top:9px}.miner-kryptex-grid>div{background:#101214;border:1px solid #292d31;border-radius:7px;padding:8px}.miner-kryptex-grid span,.miner-kryptex-grid small{display:block;color:#8f969c;font-size:10px}.miner-kryptex-grid b{display:block;color:#eee;font-size:16px;margin:3px 0}
.miner-log-wrap{margin-top:10px;background:#101010;border:1px solid #303030;border-radius:8px;padding:8px}.miner-log-head{display:flex;justify-content:space-between;align-items:center;color:#888;font-size:10px}.miner-log-head button{padding:4px 8px;font-size:10px}.miner-log-wrap pre{margin:7px 0 0;max-height:330px;overflow:auto;white-space:pre-wrap;word-break:break-word;font-size:10px;line-height:1.35;color:#cfcfcf}
@media(max-width:1100px){.miner-config{grid-template-columns:repeat(2,minmax(150px,1fr))}.miner-summary{grid-template-columns:repeat(2,minmax(150px,1fr))}.miner-kryptex-grid{grid-template-columns:repeat(2,minmax(140px,1fr))}}
'''
    dashboard = dashboard.replace("</style>", css + "\n</style>", 1)

    # Tab switching is finalized centrally in ai6_monitor_auto.py.

    js = r'''
let minerConfigLoaded=false;

function minerFmtUptime(s){
 if(s==null)return '—';
 s=Math.floor(s);const d=Math.floor(s/86400);s%=86400;const h=Math.floor(s/3600);s%=3600;const m=Math.floor(s/60);
 return (d?d+'d ':'')+(h?h+'h ':'')+m+'m';
}

function loadMinerConfig(c,detected){
 if(minerConfigLoaded)return;
 minerName.value=c.name||'Pearl / PearlHash';
 minerPool.value=c.pool||'prl.kryptex.network:7048';
 minerWallet.value=c.wallet||'';
 minerWorker.value=c.worker||'ai6-cmp40';
 minerAlgorithm.value=c.algorithm||'pearlhash';
 minerGpu.value=c.gpu||'0';
 minerBinary.value=c.binary||detected||'';
 minerExtra.value=c.extra_args||'';
 minerTempLimit.value=c.temp_limit||80;
 minerTempResume.value=c.temp_resume||70;
 minerEnergyKzt.value=(c.energy_kzt_kwh??35);
 minerPoolFee.value=(c.pool_fee_pct??2);
 minerConfigLoaded=true;
}

async function refreshMiner(){
 try{
  const r=await fetch('/api/miner/status',{cache:'no-store'});const s=await r.json();
  if(!r.ok)throw new Error(s.detail||JSON.stringify(s));
  loadMinerConfig(s.config||{},s.binary_detected);
  minerState.textContent=s.running?'● RUNNING':'● STOPPED';
  minerState.className='status '+(s.running?'ready':'down');
  const m=s.metrics||{};
  minerHashrate.textContent=m.hashrate==null?'—':Number(m.hashrate).toFixed(2)+' '+(m.hashrate_unit||'');
  minerAlgo.textContent=(m.avg_1m==null?'':'avg 1m '+Number(m.avg_1m).toFixed(2)+' TH/s • ')+((s.config&&s.config.algorithm)||'pearlhash');
  minerShares.textContent=(m.accepted||0)+' / '+(m.stale||0)+' / '+(m.rejected||0);
  minerLatency.textContent=(m.latency_ms==null?'A / S / R':'A / S / R • '+m.latency_ms+' ms')+(m.pool_hashrate==null?'':' • pool '+Number(m.pool_hashrate).toFixed(2)+' '+(m.pool_hashrate_unit||''));
  minerUptime.textContent=minerFmtUptime(s.uptime_s);
  minerPid.textContent='PID '+(s.pid??'—');
  minerAiState.textContent=s.vllm_busy?'AI BUSY':'IDLE';
  minerAiState.className=s.vllm_busy?'bad':'ok';
  minerLogPath.textContent=s.log_path||'ForgeMiner log';
  renderMinerGpuGrid(s.gpus||[],String((s.config||{}).gpu||'0'));
  renderMinerProfitability(s.profitability||{});
  renderKryptexStats(s.kryptex||{});
 }catch(e){minerMsg.textContent='Status error: '+e.message;}
}

function minerNum(v,d=2){return (v==null||!Number.isFinite(Number(v)))?'—':Number(v).toFixed(d);}
function minerMoney(v){return (v==null||!Number.isFinite(Number(v)))?'—':Math.round(Number(v)).toLocaleString('ru-RU');}
function minerHash(v){
 if(v==null||!Number.isFinite(Number(v)))return '—';
 const n=Number(v); if(n>=1e18)return (n/1e18).toFixed(2)+' EH/s';
 if(n>=1e15)return (n/1e15).toFixed(2)+' PH/s';
 if(n>=1e12)return (n/1e12).toFixed(2)+' TH/s';
 if(n>=1e9)return (n/1e9).toFixed(2)+' GH/s';
 return n.toFixed(0)+' H/s';
}
function renderMinerProfitability(p){
 const m=p.market||{};
 minerEfficiency.textContent=p.efficiency_hashrate_per_w==null?'—':Number(p.efficiency_hashrate_per_w).toFixed(3)+' TH/s/W';
 minerEfficiencySub.textContent='current GPU hashrate / '+minerNum(p.power_w,1)+' W';
 minerPrlPrice.textContent=m.prl_usdt==null?'—':'$'+Number(m.prl_usdt).toFixed(4);
 minerPriceSource.textContent=(m.price_source||'price unavailable')+(m.usd_kzt?' • USD/KZT '+Number(m.usd_kzt).toFixed(1):'');
 minerNetwork.textContent=minerHash(m.network_hashrate_hs);
 minerDifficulty.textContent='difficulty '+(m.difficulty==null?'—':Number(m.difficulty).toLocaleString('en-US'))+(m.block_time_s?' • block ~'+Math.round(m.block_time_s)+'s':'');
 minerElectricity.textContent=minerMoney(p.electricity_kzt_day)+' ₸/day';
 minerEnergyRate.textContent=minerNum(p.energy_kzt_kwh,1)+' ₸/kWh • GPU only';
 profitPrlHour.textContent=minerNum(p.prl_hour,4);
 profitPrlDay.textContent=minerNum(p.prl_day,3);
 profitPrlMonth.textContent=minerNum(p.prl_month,2);
 profitGrossHour.textContent=minerMoney(p.gross_kzt_hour);
 profitGrossDay.textContent=minerMoney(p.gross_kzt_day);
 profitGrossMonth.textContent=minerMoney(p.gross_kzt_month);
 profitElecHour.textContent=minerMoney(p.electricity_kzt_hour);
 profitElecDay.textContent=minerMoney(p.electricity_kzt_day);
 profitElecMonth.textContent=minerMoney(p.electricity_kzt_month);
 profitNetHour.textContent=minerMoney(p.net_kzt_hour);
 profitNetDay.textContent=minerMoney(p.net_kzt_day);
 profitNetMonth.textContent=minerMoney(p.net_kzt_month);
 const src=[m.price_source,m.fx_source,m.source].filter(Boolean).join(' • ');
 minerProfitNote.textContent='Estimate from current miner avg1m, live PRL price/network data, pool fee '+minerNum(p.pool_fee_pct,1)+'%, and GPU-only electricity. '+(src?'Sources: '+src+'. ':'')+'Actual payout varies with pool luck, difficulty and price.';
}

function renderKryptexStats(k){
 kryptexStatsLink.href=k.url||'#';
 kryptexPayoutsLink.href=k.payouts_url||'#';
 kryptexSettingsLink.href=k.settings_url||'#';
 kxWorkers.textContent=(k.workers_online==null?'—':k.workers_online)+' / '+(k.workers_offline==null?'—':k.workers_offline);
 kxHash30m.textContent=k.hashrate_30m_ths==null?'—':Number(k.hashrate_30m_ths).toFixed(2)+' TH/s';
 kxHashLong.textContent='3h '+(k.hashrate_3h_ths==null?'—':Number(k.hashrate_3h_ths).toFixed(2)+' TH/s')+' • 24h '+(k.hashrate_24h_ths==null?'—':Number(k.hashrate_24h_ths).toFixed(2)+' TH/s');
 kxBalance.textContent=k.confirmed_prl==null?'—':Number(k.confirmed_prl).toFixed(6)+' PRL';
 kxPending.textContent='pending '+(k.pending_prl==null?'—':Number(k.pending_prl).toFixed(6)+' PRL');
 kxPaid.textContent=k.total_paid_prl==null?'—':Number(k.total_paid_prl).toFixed(6)+' PRL';
 kxRewards.textContent='7d '+(k.reward_7d_prl==null?'—':Number(k.reward_7d_prl).toFixed(6))+' • 30d '+(k.reward_30d_prl==null?'—':Number(k.reward_30d_prl).toFixed(6));
 kxShares.textContent=(k.worker_valid==null?'—':k.worker_valid)+' / '+(k.worker_stale==null?'—':k.worker_stale)+' / '+(k.worker_invalid==null?'—':k.worker_invalid);
 kxWorker.textContent=(k.worker_name||'—')+(k.worker_mode?' • '+k.worker_mode:'');
 kxMiner.textContent=k.worker_miner||'miner —';
 kxError.textContent=k.error?'Kryptex fetch error: '+k.error:'';
 if(k.updated_at){
   const d=new Date(Number(k.updated_at)*1000);
   kryptexUpdated.textContent='public miner page • '+d.toLocaleTimeString()+' • refresh ~60s';
 }else{
   kryptexUpdated.textContent='public miner page • refresh ~60s';
 }
}

function renderMinerGpuGrid(gpus,gpuString){
 const sel=new Set(String(gpuString||'').split(',').map(x=>Number(x.trim())).filter(Number.isFinite));
 minerGpuGrid.innerHTML=(gpus||[]).map(g=>{
  const active=sel.has(g.index);
  const load=Number(g.load_pct||0),p=Number(g.power_w||0),pl=Number(g.power_limit_w||0),t=Number(g.temp_c||0);
  const loadCls=load>=90?'ok':load>=50?'warn':'';
  const pCls=pl&&p/pl>=.98?'bad':pl&&p/pl>=.90?'warn':'ok';
  const tCls=t>=80?'bad':t>=70?'warn':'ok';
  const used=Number(g.vram_used_mib||0)/1024,total=Number(g.vram_total_mib||0)/1024;
  return '<div class="vllm-gpu-card'+(active?' selected':'')+'">'+
   '<div class="vllm-gpu-card-head"><b>#'+g.index+' '+escHtml(g.name||'GPU')+'</b><span>'+(active?'miner selected':'available')+'</span></div>'+
   '<div class="vllm-gpu-stats">'+
    '<div>Load<b class="'+loadCls+'">'+load.toFixed(0)+'%</b></div>'+
    '<div>Power<b class="'+pCls+'">'+p.toFixed(1)+' W</b><span> / '+pl.toFixed(0)+' W</span></div>'+
    '<div>Temp<b class="'+tCls+'">'+t.toFixed(0)+'°C</b><span> fan '+Number(g.fan_pct||0).toFixed(0)+'%</span></div>'+
    '<div>VRAM<b>'+used.toFixed(2)+' / '+total.toFixed(2)+' GiB</b></div>'+
   '</div></div>';
 }).join('')||'<div class="muted">No NVIDIA GPUs detected</div>';
}

async function saveMinerConfig(){
 const payload={
  name:minerName.value.trim(),algorithm:minerAlgorithm.value.trim(),pool:minerPool.value.trim(),
  wallet:minerWallet.value.trim(),worker:minerWorker.value.trim(),gpu:minerGpu.value.trim(),
  binary:minerBinary.value.trim(),extra_args:minerExtra.value.trim(),
  temp_limit:Number(minerTempLimit.value),temp_resume:Number(minerTempResume.value),
  energy_kzt_kwh:Number(minerEnergyKzt.value),pool_fee_pct:Number(minerPoolFee.value)
 };
 minerMsg.textContent='Saving…';
 try{
  const r=await fetch('/api/miner/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  minerMsg.textContent='Config saved'+(d.binary_detected?' • '+d.binary_detected:'');
 }catch(e){minerMsg.textContent='Save error: '+e.message;}
}

async function controlMiner(action){
 if(action==='start'){
  await saveMinerConfig();
  if(!confirm('Start ForgeMiner on selected GPU(s)?'))return;
  try{
   await setMinerPowerLimit(true);
  }catch(e){
   minerMsg.textContent='Start blocked: '+e.message;
   return;
  }
 }else if(!confirm('Stop ForgeMiner?'))return;
 minerMsg.textContent=action==='start'?'Starting miner…':'Stopping miner…';
 try{
  const r=await fetch('/api/miner/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})});
  const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  minerMsg.textContent=d.message||'OK';
  setTimeout(refreshMiner,700);setTimeout(refreshMinerLogs,1200);
 }catch(e){minerMsg.textContent='Miner error: '+e.message;}
}

async function setMinerPowerLimit(throwOnError=false){
 const gpu=Number(minerPlGpu.value),watts=Number(minerPlPreset.value);
 minerPlMsg.textContent='Setting '+watts+' W…';
 try{
  const r=await fetch('/api/gpu/power-limit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({gpu,watts})});
  const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  minerPlMsg.textContent='GPU'+gpu+' PL '+Number(d.current_w).toFixed(0)+' W';
  await refreshMiner();
  return d;
 }catch(e){
  minerPlMsg.textContent='PL error: '+e.message;
  if(throwOnError)throw e;
  return null;
 }
}

async function refreshMinerLogs(){
 try{
  const r=await fetch('/api/miner/logs',{cache:'no-store'});
  const t=await r.text();
  minerLogText.textContent=t||'(empty)';
  minerLogText.scrollTop=minerLogText.scrollHeight;
 }catch(e){minerLogText.textContent='Log error: '+e.message;}
}

setInterval(()=>{if(document.getElementById('minerTab')&&document.getElementById('minerTab').style.display!=='none')refreshMiner();},2000);
'''
    dashboard = dashboard.replace("refresh();setInterval(refresh,2000);", js + "\nrefresh();setInterval(refresh,2000);", 1)

    base.DASHBOARD = dashboard
    dynamic.base.DASHBOARD = dashboard
    dynamic.app = base.app

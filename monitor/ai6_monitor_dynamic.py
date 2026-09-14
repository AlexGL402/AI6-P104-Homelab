#!/usr/bin/env python3
"""Dynamic worker discovery layer for the AI6 monitor.

Keeps the existing dashboard/API/host controls, but replaces the hard-coded
LLM worker list with live discovery of local llama-server processes.

Discovery pipeline (see README_RU.md for user-facing details):
1. Scan all host processes; a process counts as a worker if its name or
   command line contains `llama-server`.
2. Parse CLI args (`--port`, `--model`, `--alias`, `--ctx-size`, both
   `--flag value` and `--flag=value` forms). No port -> process skipped.
3. Enrich each worker: health state, live model id, GPU assignment from
   process env, PID/uptime, and last gen/prompt tok/s parsed from the
   journal (managed) or the process stdout file (manual).
"""

import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

import psutil

import ai6_monitor as base


_ORIGINAL_COLLECT_STATS = base.collect_stats

# Journal/log tail length used when extracting the last measured tok/s.
_LOG_TAIL_LINES = 160

# Timeout (seconds) for external commands used while enriching a worker.
_SUBPROCESS_TIMEOUT = 1.5

# /v1/models is only polled for ready workers; keep the probe short.
_MODELS_TIMEOUT = 0.7

# Arguments we understand when parsing a llama-server command line.
_CLI_OPTIONS = {
    "--port": "port",
    "-p": "port",
    "--ctx-size": "ctx",
    "-c": "ctx",
    "--alias": "alias",
    "--model": "model",
    "-m": "model",
}


def _cli_options(cmdline):
    """Parse a llama-server argv into {option: value}, handling both
    `--flag value` and `--flag=value` forms."""
    cmdline = tuple(cmdline)
    options = {}
    for i, arg in enumerate(cmdline):
        key = _CLI_OPTIONS.get(arg)
        if key is not None and i + 1 < len(cmdline):
            options.setdefault(key, cmdline[i + 1])
        for flag, key in _CLI_OPTIONS.items():
            prefix = flag + "="
            if arg.startswith(prefix):
                options.setdefault(key, arg[len(prefix):])
    return options


def _parse_int(value, lo=None, hi=None):
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if lo is not None and number < lo:
        return None
    if hi is not None and number > hi:
        return None
    return number


def _process_environment(proc):
    """Best-effort read of a process environment (may be denied for other users)."""
    try:
        return proc.environ()
    except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
        return {}


def _gpu_assignment(proc):
    env = _process_environment(proc)
    raw = env.get("CUDA_VISIBLE_DEVICES") or env.get("NVIDIA_VISIBLE_DEVICES")
    if not raw or raw.lower() in ("all", "void", "none"):
        return [], "GPU auto/all"
    devices = [x.strip() for x in raw.split(",") if x.strip()]
    return devices, ("GPU " + "+".join(devices)) if devices else "GPU auto/all"


def _stdout_log_path(pid):
    try:
        target = os.readlink(f"/proc/{pid}/fd/1")
    except OSError:
        return None
    if target.startswith("/") and Path(target).is_file():
        return target
    return None


def _tail_text(path, lines=_LOG_TAIL_LINES):
    if not path:
        return ""
    try:
        p = subprocess.run(
            ["tail", "-n", str(lines), path], capture_output=True, text=True,
            timeout=_SUBPROCESS_TIMEOUT,
        )
        return p.stdout or ""
    except Exception:
        return ""


def _extract_timings(text):
    gen = prompt = None
    prompt_tokens = None
    for line in reversed(text.splitlines()):
        if gen is None and "eval time" in line and "tokens per second" in line and "prompt eval time" not in line:
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s+tokens per second", line)
            if m:
                gen = float(m.group(1))
        if prompt is None and "prompt eval time" in line:
            m = re.search(r"/\s+(\d+)\s+tokens.*?([0-9]+(?:\.[0-9]+)?)\s+tokens per second", line)
            if m:
                prompt_tokens = int(m.group(1))
                prompt = float(m.group(2))
        if gen is not None and prompt is not None:
            break
    return gen, prompt, prompt_tokens


def _is_llama_server(proc):
    info = proc.info or {}
    name = info.get("name") or ""
    joined = " ".join(info.get("cmdline") or [])
    return "llama-server" in name or "llama-server" in joined


def _worker_processes():
    found = {}
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            if not _is_llama_server(proc):
                continue
            port = _parse_int(_cli_options(proc.info.get("cmdline") or []).get("port"), 1, 65535)
            if port is None:
                continue
            found[port] = proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return dict(sorted(found.items()))


def _dynamic_worker_details(port, proc):
    active = base.service_ok(port)
    state, status_text = base._health_state(port, active)
    options = _cli_options(proc.info.get("cmdline") or [])
    alias = options.get("alias")
    model_path = options.get("model")
    ctx = _parse_int(options.get("ctx"))
    gpu_devices, gpu_label = _gpu_assignment(proc)
    service = base._service_for_port(port)

    info = {
        "active": active,
        "ready": state == "ready",
        "state": state,
        "status_text": status_text,
        "model": alias or (Path(model_path).name if model_path else None),
        "model_path": model_path,
        "pid": proc.pid,
        "port": port,
        "ctx": ctx,
        "gpu_devices": gpu_devices,
        "gpu_label": gpu_label,
        "service": service,
        "managed": bool(service),
        "last_tok_s": None,
        "last_prompt_tok_s": None,
        "last_prompt_tokens": None,
        "uptime_s": max(0, int(time.time() - (proc.info.get("create_time") or time.time()))),
    }

    if state == "ready":
        live_id = _live_model_id(port)
        if live_id:
            info["model"] = live_id

    text = ""
    if service:
        try:
            p = subprocess.run(
                ["journalctl", "-u", service, "-n", str(_LOG_TAIL_LINES), "--no-pager", "-o", "cat"],
                capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT,
            )
            text = p.stdout or ""
        except Exception:
            pass
    if not text:
        info["log_path"] = _stdout_log_path(proc.pid)
        text = _tail_text(info.get("log_path"))

    gen, prompt, prompt_tokens = _extract_timings(text)
    info["last_tok_s"] = gen
    info["last_prompt_tok_s"] = prompt
    info["last_prompt_tokens"] = prompt_tokens
    return info


def _live_model_id(port):
    """Return the live model id from /v1/models, or None if unavailable."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=_MODELS_TIMEOUT) as r:
            data = json.load(r)
        models = data.get("data") or []
        if models and models[0].get("id"):
            return models[0]["id"]
    except Exception:
        pass
    return None


def discover_workers():
    return {
        str(port): _dynamic_worker_details(port, proc)
        for port, proc in _worker_processes().items()
    }


def collect_stats():
    stats = _ORIGINAL_COLLECT_STATS()
    workers = discover_workers()
    llama = {
        "profile": base.worker_profile(),
        "discovery": "process",
        "count": len(workers),
        "workers": workers,
    }
    for port, info in workers.items():
        llama[port] = bool(info.get("active"))
    stats["llama"] = llama
    return stats


base.collect_stats = collect_stats


# Patch only the worker-rendering pieces of the existing dashboard so the rest
# of the UI (GPU telemetry, PSU logging, terminal and host controls) stays intact.
_dashboard = base.DASHBOARD
_dashboard = _dashboard.replace(
    "Model, GPU assignment, readiness and last measured throughput",
    "Auto-discovered llama.cpp processes: model, GPUs, context, readiness and throughput",
)

_worker_js = r'''function dynamicWorkerCard(p,w){
 const model=w.model||'llama-server';
 const gen=w.last_tok_s!=null?w.last_tok_s.toFixed(2):'—';
 const prompt=w.last_prompt_tok_s!=null?w.last_prompt_tok_s.toFixed(1):'—';
 const promptSize=w.last_prompt_tokens!=null?w.last_prompt_tokens:'—';
 const up=w.uptime_s!=null?fmtUptime(w.uptime_s):'—';
 const gpu=w.gpu_label||'GPU auto/all';
 const ctx=w.ctx!=null?w.ctx.toLocaleString():'—';
 const meta=gpu+' • ctx '+ctx+' • PID '+(w.pid??'—');
 const actions=w.managed
   ? '<div class="worker-actions"><button onclick="workerAction(\\'start\\','+p+')">Start</button><button onclick="workerAction(\\'stop\\','+p+')">Stop</button><button onclick="workerAction(\\'restart\\','+p+')">Restart</button></div>'
   : '<div class="test-note">Manual process • start/stop from terminal</div>';
 return '<div class="worker-card"><div class="worker-head"><span class="worker-port">'+p+'</span><span class="status '+(w.state||'down')+'">'+stateIcon(w)+' '+(w.status_text||'Down')+'</span></div><div class="worker-model" title="'+model+'">'+model+'</div><div class="worker-meta">'+meta+'</div><div class="worker-stats"><div class="worker-stat">Gen<b>'+gen+' tok/s</b></div><div class="worker-stat">Prompt<b>'+prompt+' tok/s</b></div><div class="worker-stat">Prompt size<b>'+promptSize+'</b></div><div class="worker-stat">Uptime<b>'+up+'</b></div></div>'+actions+'</div>';
}

async function refresh(){'''

_dashboard, n = re.subn(
    r"function regularWorkerCard\(p,w,profile\)\{.*?async function refresh\(\)\{",
    lambda _m: _worker_js,
    _dashboard,
    count=1,
    flags=re.S,
)
if n != 1:
    raise RuntimeError("AI6 monitor dashboard worker function patch did not match")

_refresh_workers = r''' const ws=s.llama.workers||{};const ports=Object.keys(ws).map(Number).sort((a,b)=>a-b);
 const readyCount=ports.filter(p=>(ws[String(p)]||{}).state==='ready').length;const loadingCount=ports.filter(p=>['loading','starting'].includes((ws[String(p)]||{}).state)).length;const downCount=ports.length-readyCount-loadingCount;
 workers.innerHTML='<span class="ready">'+readyCount+'</span>/<span class="loading">'+loadingCount+'</span>/<span class="down">'+downCount+'</span>';
 profile.innerHTML='auto-discovery • '+ports.length+' llama-server'+(ports.length===1?'':'s')+' • managed profile '+(s.llama.profile==='222'?'2+2+2':'3+3');
 workerbuttons.innerHTML=ports.length?ports.map(p=>dynamicWorkerCard(p,ws[String(p)]||{})).join(''):'<div class="muted">No llama-server processes discovered</div>';
'''

_dashboard, n = re.subn(
    r" const ports=s\.llama\.profile==='222'.*?workerbuttons\.innerHTML=.*?;\n",
    lambda _m: _refresh_workers,
    _dashboard,
    count=1,
    flags=re.S,
)
if n != 1:
    raise RuntimeError("AI6 monitor dashboard refresh patch did not match")

base.DASHBOARD = _dashboard
app = base.app

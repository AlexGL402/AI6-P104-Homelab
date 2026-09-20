#!/usr/bin/env python3
"""vLLM control/status tab for the AI6 Host Monitor."""

import concurrent.futures
import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from importlib.metadata import version as package_version, PackageNotFoundError

import psutil
from pydantic import BaseModel, Field
from fastapi.responses import PlainTextResponse

import ai6_monitor_dynamic as dynamic

base = dynamic.base

_VLLM_ENV = Path(os.environ.get("AI6_VLLM_ENV", str(Path.home() / "vllm-env")))
_VLLM_BIN = Path(os.environ.get("AI6_VLLM_BIN", str(_VLLM_ENV / "bin/vllm")))
_VLLM_MODELS_DIR = Path(os.environ.get("AI6_VLLM_MODELS_DIR", str(Path.home() / "models/vllm")))
_VLLM_LOG_DIR = Path(os.environ.get("AI6_VLLM_LOG_DIR", str(Path.home() / ".local/state/ai6-monitor/vllm")))
_VLLM_DEFAULT_PORT = int(os.environ.get("AI6_VLLM_PORT", "8012"))
_BENCH_RESULTS = []
_LAST_PROMPT_RESULT = None
_LOAD_TIMELINE = {}

try:
    _VLLM_VERSION = package_version("vllm")
except PackageNotFoundError:
    _VLLM_VERSION = "unknown"


class VllmStartCommand(BaseModel):
    model: str
    port: int = Field(default=_VLLM_DEFAULT_PORT, ge=1024, le=65535)
    max_model_len: int = Field(default=4096, ge=256, le=262144)
    gpu_memory_utilization: float = Field(default=0.90, ge=0.10, le=0.99)
    enforce_eager: bool = False
    dtype: str = "half"


class VllmControlCommand(BaseModel):
    action: str


class VllmBenchmarkCommand(BaseModel):
    concurrency: int = Field(ge=1, le=128)
    max_tokens: int = Field(default=512, ge=16, le=4096)


class VllmPromptCommand(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)
    max_tokens: int = Field(default=512, ge=16, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


def _vllm_process():
    candidates = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time", "username"]):
        try:
            cmd = proc.info.get("cmdline") or []
            joined = " ".join(cmd)
            if "vllm" not in joined.lower() or " serve " not in (" " + joined + " ").lower():
                continue
            candidates.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.info.get("create_time") or 0)[0]


def _arg_value(cmdline, flag, default=None):
    for i, arg in enumerate(cmdline):
        if arg == flag and i + 1 < len(cmdline):
            return cmdline[i + 1]
        prefix = flag + "="
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return default


def _model_quantization(model_path):
    if not model_path:
        return None
    try:
        cfg = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
        q = cfg.get("quantization_config") or {}
        method = q.get("quant_method") or q.get("quantization_method")
        if method:
            return str(method).upper()
        name = Path(model_path).name.upper()
        for token in ("AWQ", "GPTQ", "FP8", "INT8", "INT4"):
            if token in name:
                return token
    except Exception:
        pass
    return None


def _validate_vllm_model(model_text):
    model = Path(model_text).resolve()
    try:
        model.relative_to(_VLLM_MODELS_DIR.resolve())
    except Exception:
        raise base.HTTPException(status_code=400, detail="model must be below vLLM models directory")
    if not model.is_dir() or not (model / "config.json").is_file():
        raise base.HTTPException(status_code=400, detail="vLLM model directory/config.json not found")
    return model


def _http_json(url, timeout=0.8):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    except Exception:
        return None


def _http_text(url, timeout=0.8):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return ""


def _metric_value(text, metric):
    pattern = re.compile(r"^" + re.escape(metric) + r"(?:\{[^}]*\})?\s+([-+0-9.eE]+)\s*$", re.M)
    m = pattern.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _vllm_status():
    global _LOAD_TIMELINE
    proc = _vllm_process()
    if proc is None:
        _LOAD_TIMELINE = {}
        return {
            "running": False,
            "ready": False,
            "pid": None,
            "port": _VLLM_DEFAULT_PORT,
            "model": None,
            "model_path": None,
            "uptime_s": 0,
            "metrics": {},
        }

    cmd = proc.info.get("cmdline") or []
    port = int(_arg_value(cmd, "--port", str(_VLLM_DEFAULT_PORT)))
    model_path = None
    try:
        serve_i = cmd.index("serve")
        if serve_i + 1 < len(cmd):
            model_path = cmd[serve_i + 1]
    except ValueError:
        pass

    health_ok = False
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.7) as r:
            health_ok = 200 <= r.status < 300
    except Exception:
        pass

    model_id = None
    models = _http_json(f"http://127.0.0.1:{port}/v1/models")
    if models:
        entries = models.get("data") or []
        if entries:
            model_id = entries[0].get("id")

    # Approximate startup-stage timing. Status is polled by the UI, so milestones
    # are sampled at the dashboard polling cadence rather than instrumented inside vLLM.
    now_mono = time.monotonic()
    if _LOAD_TIMELINE.get("pid") != proc.pid:
        _LOAD_TIMELINE = {"pid": proc.pid, "process_seen": now_mono}
    try:
        gpu0 = (base.gpu_stats() or [None])[0] or {}
        used_gib = float(gpu0.get("memory_used_mib") or 0) / 1024.0
        for key, threshold in (("weights_started", 0.2), ("weights_mid", 2.0), ("engine_init", 5.0), ("warmup", 6.0)):
            if used_gib >= threshold and key not in _LOAD_TIMELINE:
                _LOAD_TIMELINE[key] = now_mono
    except Exception:
        pass
    if health_ok and "ready" not in _LOAD_TIMELINE:
        _LOAD_TIMELINE["ready"] = now_mono

    metrics_text = _http_text(f"http://127.0.0.1:{port}/metrics")
    metrics = {
        "generation_tokens_total": _metric_value(metrics_text, "vllm:generation_tokens_total"),
        "prompt_tokens_total": _metric_value(metrics_text, "vllm:prompt_tokens_total"),
        "running": _metric_value(metrics_text, "vllm:num_requests_running"),
        "waiting": _metric_value(metrics_text, "vllm:num_requests_waiting"),
        "kv_cache_usage": _metric_value(metrics_text, "vllm:gpu_cache_usage_perc"),
    }

    return {
        "running": True,
        "ready": health_ok,
        "pid": proc.pid,
        "port": port,
        "model": model_id or (Path(model_path).name if model_path else None),
        "model_path": model_path,
        "uptime_s": max(0, int(time.time() - (proc.info.get("create_time") or time.time()))),
        "max_model_len": int(_arg_value(cmd, "--max-model-len", "0") or 0) or None,
        "gpu_memory_utilization": float(_arg_value(cmd, "--gpu-memory-utilization", "0") or 0) or None,
        "enforce_eager": "--enforce-eager" in cmd,
        "metrics": metrics,
        "load_timeline": _load_timeline_summary(),
        "quantization": _model_quantization(model_path),
        "vllm_version": _VLLM_VERSION,
    }


def _load_timeline_summary():
    if not _LOAD_TIMELINE:
        return {}
    t0 = _LOAD_TIMELINE.get("process_seen")
    if t0 is None:
        return {}
    out = {}
    for key in ("weights_started", "weights_mid", "engine_init", "warmup", "ready"):
        if key in _LOAD_TIMELINE:
            out[key + "_s"] = round(_LOAD_TIMELINE[key] - t0, 2)
    if "ready" in _LOAD_TIMELINE:
        out["total_startup_s"] = round(_LOAD_TIMELINE["ready"] - t0, 2)
    out["sampled"] = True
    return out


def _stop_vllm():
    proc = _vllm_process()
    if proc is None:
        return
    try:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        try:
            proc.wait(timeout=10)
        except psutil.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
    except psutil.NoSuchProcess:
        pass


def _start_vllm(cmd):
    if _vllm_process() is not None:
        raise base.HTTPException(status_code=409, detail="vLLM is already running")
    if not _VLLM_BIN.is_file():
        raise base.HTTPException(status_code=409, detail=f"vLLM binary not found: {_VLLM_BIN}")

    model = _validate_vllm_model(cmd.model)
    if base.service_ok(cmd.port):
        raise base.HTTPException(status_code=409, detail=f"port {cmd.port} is already in use")

    args = [
        str(_VLLM_BIN), "serve", str(model),
        "--host", "0.0.0.0",
        "--port", str(cmd.port),
        "--dtype", cmd.dtype,
        "--gpu-memory-utilization", f"{cmd.gpu_memory_utilization:.3f}",
        "--max-model-len", str(cmd.max_model_len),
    ]
    if cmd.enforce_eager:
        args.append("--enforce-eager")

    _VLLM_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _VLLM_LOG_DIR / f"vllm-{cmd.port}.log"
    log = log_path.open("ab", buffering=0)
    try:
        proc = subprocess.Popen(
            args,
            cwd=str(Path.home()),
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    except Exception as e:
        log.close()
        raise base.HTTPException(status_code=409, detail=f"failed to start vLLM: {e}")
    log.close()
    return {"pid": proc.pid, "port": cmd.port, "log": str(log_path), "model": str(model)}


@base.app.get("/api/vllm/models")
def api_vllm_models():
    models = []
    if _VLLM_MODELS_DIR.exists():
        for cfg in sorted(_VLLM_MODELS_DIR.rglob("config.json")):
            path = cfg.parent
            try:
                rel = path.relative_to(_VLLM_MODELS_DIR)
            except ValueError:
                rel = path.name
            models.append({"label": str(rel), "path": str(path)})
    return {"models": models}


@base.app.get("/api/vllm/status")
def api_vllm_status():
    result = _vllm_status()
    result["benchmarks"] = list(_BENCH_RESULTS[-20:])
    return result


@base.app.post("/api/vllm/start")
def api_vllm_start(cmd: VllmStartCommand):
    result = _start_vllm(cmd)
    return {"ok": True, **result}


@base.app.post("/api/vllm/control")
def api_vllm_control(cmd: VllmControlCommand):
    if cmd.action not in ("stop", "restart"):
        raise base.HTTPException(status_code=400, detail="action must be stop or restart")
    status = _vllm_status()
    if not status["running"]:
        return {"ok": True, "message": "vLLM already stopped"}
    if cmd.action == "stop":
        _stop_vllm()
        return {"ok": True}

    model = status.get("model_path")
    if not model:
        raise base.HTTPException(status_code=409, detail="cannot determine current vLLM model")
    start_cmd = VllmStartCommand(
        model=model,
        port=status.get("port") or _VLLM_DEFAULT_PORT,
        max_model_len=status.get("max_model_len") or 4096,
        gpu_memory_utilization=status.get("gpu_memory_utilization") or 0.90,
        enforce_eager=bool(status.get("enforce_eager")),
    )
    _stop_vllm()
    time.sleep(0.5)
    result = _start_vllm(start_cmd)
    return {"ok": True, **result}


def _bench_one(port, model, prompt, max_tokens):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.load(r)
    elapsed = time.perf_counter() - started
    tokens = int((data.get("usage") or {}).get("completion_tokens") or 0)
    return tokens, elapsed


@base.app.post("/api/vllm/prompt")
def api_vllm_prompt(cmd: VllmPromptCommand):
    global _LAST_PROMPT_RESULT
    status = _vllm_status()
    if not status["ready"]:
        raise base.HTTPException(status_code=409, detail="vLLM server is not ready")
    payload = json.dumps({
        "model": status["model"],
        "messages": [{"role": "user", "content": cmd.prompt}],
        "temperature": cmd.temperature,
        "max_tokens": cmd.max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{status['port']}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.load(r)
    except Exception as e:
        raise base.HTTPException(status_code=502, detail=f"vLLM request failed: {e}")
    elapsed = time.perf_counter() - started
    choice = ((data.get("choices") or [{}])[0].get("message") or {})
    usage = data.get("usage") or {}
    out_tokens = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    result = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": out_tokens,
        "elapsed_s": round(elapsed, 3),
        "output_tok_s": round(out_tokens / elapsed, 2) if elapsed > 0 and out_tokens else 0,
        "total_tok_s": round((prompt_tokens + out_tokens) / elapsed, 2) if elapsed > 0 else 0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _LAST_PROMPT_RESULT = result
    return {
        "ok": True,
        "content": choice.get("content") or "",
        "usage": usage,
        **result,
    }


@base.app.get("/api/vllm/logs")
def api_vllm_logs():
    status = _vllm_status()
    port = status.get("port") or _VLLM_DEFAULT_PORT
    log_path = _VLLM_LOG_DIR / f"vllm-{port}.log"
    if not log_path.is_file():
        return {"ok": True, "path": str(log_path), "log": "No vLLM log file yet."}
    try:
        p = subprocess.run(
            ["tail", "-n", "220", str(log_path)],
            capture_output=True, text=True, timeout=2.0,
        )
        text = p.stdout or p.stderr or ""
    except Exception as e:
        text = f"Failed to read log: {e}"
    return {"ok": True, "path": str(log_path), "log": text}


@base.app.get("/api/vllm/report")
def api_vllm_report():
    status = _vllm_status()
    stats = base.collect_stats()
    gpu = ((stats.get("gpu") or {}).get("devices") or [None])[0] or {}
    lines = [
        "# AI6 vLLM report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        "",
        "## Server",
        f"- State: {'READY' if status.get('ready') else 'LOADING' if status.get('running') else 'STOPPED'}",
        f"- PID: {status.get('pid') or '-'}",
        f"- Port: {status.get('port') or _VLLM_DEFAULT_PORT}",
        f"- Model: {status.get('model') or '-'}",
        f"- Context: {status.get('max_model_len') or '-'}",
        f"- GPU memory utilization: {status.get('gpu_memory_utilization') or '-'}",
        f"- Enforce eager: {bool(status.get('enforce_eager'))}",
        "",
        "## Startup stages (sampled)",
    ]
    timeline = status.get("load_timeline") or {}
    if timeline:
        stage_labels = [
            ("weights_started_s", "Weights started"),
            ("weights_mid_s", "Weights ~mid-load"),
            ("engine_init_s", "Engine init"),
            ("warmup_s", "Warmup"),
            ("ready_s", "API ready"),
            ("total_startup_s", "Total startup"),
        ]
        for key, label in stage_labels:
            if key in timeline:
                lines.append(f"- {label}: {timeline[key]:.2f} s from process detection")
        lines.append("- Note: startup milestones are sampled by monitor polling / VRAM thresholds, not exact vLLM byte-progress.")
    else:
        lines.append("- No startup timeline captured in this monitor session.")
    lines += [
        "",
        "## Last interactive prompt",
    ]
    if _LAST_PROMPT_RESULT:
        p = _LAST_PROMPT_RESULT
        lines += [
            f"- Prompt tokens: {p['prompt_tokens']}",
            f"- Output tokens: {p['completion_tokens']}",
            f"- Wall time: {p['elapsed_s']:.3f} s",
            f"- Output throughput: {p['output_tok_s']:.2f} tok/s",
            f"- Total token throughput: {p['total_tok_s']:.2f} tok/s",
        ]
    else:
        lines.append("- No interactive prompt run recorded in this monitor session.")
    lines += [
        "",
        "## Host / GPU snapshot",
        f"- CPU: {(stats.get('cpu') or {}).get('usage_pct', '-')} %",
        f"- RAM: {(stats.get('memory') or {}).get('usage_pct', '-')} %",
        f"- GPU: {gpu.get('name', '-')}",
        f"- GPU load: {gpu.get('utilization_pct', '-')} %",
        f"- GPU power: {gpu.get('power_w', '-')} W / limit {gpu.get('power_limit_w', '-')} W",
        f"- VRAM: {gpu.get('memory_used_mib', '-')} MiB / {gpu.get('memory_total_mib', '-')} MiB",
        f"- GPU temp: {gpu.get('temperature_c', '-')} C",
        f"- Fan: {gpu.get('fan_pct', '-')} %",
        "",
        "## UI benchmark results",
        "",
        "| Concurrent | Output/request | Total tokens | Wall s | Aggregate tok/s | Per-request tok/s |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    if _BENCH_RESULTS:
        for x in _BENCH_RESULTS:
            lines.append(
                f"| {x['concurrency']} | {x['max_tokens']} | {x['total_tokens']} | "
                f"{x['wall_s']:.3f} | {x['aggregate_tok_s']:.2f} | "
                f"{x['per_request_min_tok_s']:.2f}-{x['per_request_max_tok_s']:.2f} |"
            )
    else:
        lines.append("| - | - | - | - | - | no UI benchmark runs yet |")
    lines += [
        "",
        "## Paths",
        f"- vLLM binary: {_VLLM_BIN}",
        f"- Models: {_VLLM_MODELS_DIR}",
        f"- Log: {_VLLM_LOG_DIR / ('vllm-' + str(status.get('port') or _VLLM_DEFAULT_PORT) + '.log')}",
        "",
    ]
    body = "\n".join(lines)
    filename = "ai6-vllm-report-" + time.strftime("%Y%m%d-%H%M%S") + ".md"
    return PlainTextResponse(
        body,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@base.app.post("/api/vllm/benchmark")
def api_vllm_benchmark(cmd: VllmBenchmarkCommand):
    status = _vllm_status()
    if not status["ready"]:
        raise base.HTTPException(status_code=409, detail="vLLM server is not ready")
    port = status["port"]
    model = status["model"]
    prompts = [
        f"Write a production-quality Python example for task {i}: async concurrency, retries, timeouts, logging, type hints, graceful shutdown, metrics, and error handling."
        for i in range(1, cmd.concurrency + 1)
    ]
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=cmd.concurrency) as pool:
        futures = [
            pool.submit(_bench_one, port, model, prompt, cmd.max_tokens)
            for prompt in prompts
        ]
        results = [f.result() for f in futures]
    wall = time.perf_counter() - started
    total_tokens = sum(x[0] for x in results)
    per_request = [x[0] / x[1] if x[1] > 0 else 0 for x in results]
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "concurrency": cmd.concurrency,
        "max_tokens": cmd.max_tokens,
        "total_tokens": total_tokens,
        "wall_s": round(wall, 3),
        "aggregate_tok_s": round(total_tokens / wall, 2) if wall > 0 else 0,
        "per_request_min_tok_s": round(min(per_request), 2) if per_request else 0,
        "per_request_max_tok_s": round(max(per_request), 2) if per_request else 0,
    }
    _BENCH_RESULTS.append(row)
    del _BENCH_RESULTS[:-20]
    return {"ok": True, "result": row}


def install():
    dashboard = base.DASHBOARD

    nav = """<div class="top-tabs">
<button id="tabMonitorBtn" class="tab-btn active" onclick="showTopTab('monitor')">Monitor</button>
<button id="tabVllmBtn" class="tab-btn" onclick="showTopTab('vllm')">vLLM</button>
</div>
<div id="monitorTab">"""
    dashboard = dashboard.replace(
        '<div class="summary-grid">',
        nav + '\n<div class="summary-grid">',
        1,
    )

    vllm_html = r'''
</div>
<div id="vllmTab" style="display:none">
  <div class="section">
    <div class="section-head">
      <div><div class="section-title">vLLM Server</div><div class="section-sub">Dedicated vLLM control, live metrics and continuous-batching benchmark</div></div>
      <div id="vllmState" class="status down">● STOPPED</div>
    </div>
    <div class="vllm-form">
      <label>Model<select id="vllmModel"></select></label>
      <label>Port<input id="vllmPort" type="number" value="8012" min="1024" max="65535"></label>
      <label>Context<input id="vllmCtx" type="number" value="4096" min="256"></label>
      <label>GPU memory<input id="vllmMem" type="number" value="0.90" min="0.10" max="0.99" step="0.01"></label>
      <label class="vllm-check"><input id="vllmEager" type="checkbox"> Enforce eager</label>
    </div>
    <div id="vllmLoadWrap" class="vllm-load-wrap" style="display:none">
      <div class="vllm-load-line"><span id="vllmLoadText">Loading model…</span><span id="vllmLoadPct">0%</span></div>
      <div class="vllm-load-bar"><div id="vllmLoadFill" class="vllm-load-fill"></div></div>
      <div id="vllmLoadSub" class="vllm-load-sub">Waiting for model initialization…</div>
    </div>
    <div class="vllm-actions">
      <button onclick="startVllm()">Start</button>
      <button onclick="controlVllm('stop')">Stop</button>
      <button onclick="controlVllm('restart')">Restart</button>
      <button onclick="downloadVllmReport()">Download report</button>
      <button onclick="toggleVllmLogs()">Logs</button>
      <span id="vllmMsg" class="muted"></span>
    </div>
    <div class="vllm-cards">
      <div class="worker-stat">PID<b id="vllmPid">—</b></div>
      <div class="worker-stat">Model<b id="vllmLiveModel">—</b></div>
      <div class="worker-stat">Uptime<b id="vllmUptime">—</b></div>
      <div class="worker-stat">Running<b id="vllmRunning">—</b></div>
      <div class="worker-stat">Waiting<b id="vllmWaiting">—</b></div>
      <div class="worker-stat">KV cache<b id="vllmKv">—</b></div>
      <div class="worker-stat">Decode live<b id="vllmDecode">—</b></div>
      <div class="worker-stat">Prompt live<b id="vllmPrompt">—</b></div>
    </div>
    <div id="vllmLogWrap" class="vllm-log-wrap" style="display:none">
      <div class="vllm-log-head"><span id="vllmLogPath">vLLM log</span><button onclick="refreshVllmLogs()">Refresh</button></div>
      <pre id="vllmLogText">Loading…</pre>
    </div>
    <div class="vllm-host-strip">
      <div class="vllm-mini">CPU<b id="vllmCpu">—</b><small id="vllmCpuSub">—</small></div>
      <div class="vllm-mini">RAM<b id="vllmRam">—</b><small id="vllmRamSub">—</small></div>
      <div class="vllm-mini">GPU load<b id="vllmGpuLoad">—</b><small id="vllmGpuName">—</small></div>
      <div class="vllm-mini">GPU power<b id="vllmGpuPower">—</b><small id="vllmGpuPowerSub">—</small></div>
      <div class="vllm-mini">VRAM<b id="vllmGpuVram">—</b><small id="vllmGpuVramSub">—</small></div>
      <div class="vllm-mini">GPU temp<b id="vllmGpuTemp">—</b><small id="vllmGpuFan">—</small></div>
    </div>
  </div>

  <div class="section">
    <div class="section-head">
      <div><div class="section-title">Test Prompt</div><div class="section-sub">Send a real coding/chat task to the running vLLM server</div></div>
    </div>
    <textarea id="vllmPromptInput" class="vllm-prompt" placeholder="Paste a test task here...">Write a production-quality Python implementation of an asynchronous HTTP crawler with retries, timeout handling, URL deduplication, SHA256 hashing, logging, graceful shutdown, and type hints.</textarea>
    <div class="vllm-prompt-actions">
      <label>Output tokens<input id="vllmPromptTokens" type="number" value="512" min="16" max="4096"></label>
      <label>Temperature<input id="vllmPromptTemp" type="number" value="0" min="0" max="2" step="0.1"></label>
      <button onclick="runVllmPrompt()">Run prompt</button>
      <button onclick="clearVllmPrompt()">Clear output</button>
      <span id="vllmPromptMsg" class="muted"></span>
    </div>
    <pre id="vllmPromptOutput" class="vllm-prompt-output">Response will appear here.</pre>
  </div>

  <div class="section">
    <div class="section-head">
      <div><div class="section-title">vLLM Batch Benchmark</div><div class="section-sub">Same server, fixed 512 output tokens per request</div></div>
    </div>
    <div class="vllm-bench-meta">
      <span>Model <b id="benchModel">—</b></span>
      <span>Quant <b id="benchQuant">—</b></span>
      <span>Context <b id="benchCtx">—</b></span>
      <span>Mode <b id="benchMode">—</b></span>
      <span>PL <b id="benchPl">—</b></span>
      <span>GPU <b id="benchGpu">—</b></span>
      <span>vLLM <b id="benchVllm">—</b></span>
    </div>
    <div class="vllm-bench-buttons">
      <button onclick="runVllmBench(1)">1</button>
      <button onclick="runVllmBench(4)">4</button>
      <button onclick="runVllmBench(8)">8</button>
      <button onclick="runVllmBench(16)">16</button>
      <button onclick="runVllmBench(24)">24</button>
      <button onclick="runVllmBench(32)">32</button>
      <button onclick="runVllmBench(40)">40</button>
      <button onclick="runVllmBench(48)">48</button>
      <button onclick="runVllmBench(64)">64</button>
      <span id="vllmBenchMsg" class="muted"></span>
    </div>
    <div class="vllm-table-wrap">
      <table class="vllm-table">
        <thead><tr><th>Concurrent</th><th>Total tokens</th><th>Wall</th><th>Aggregate</th><th>Per request</th></tr></thead>
        <tbody id="vllmBenchRows"><tr><td colspan="5" class="muted">No UI benchmark runs yet</td></tr></tbody>
      </table>
    </div>
  </div>
</div>
'''
    dashboard = dashboard.replace("<script>", vllm_html + "\n<script>", 1)

    css = r'''
.top-tabs{display:flex;gap:8px;margin:14px 0 2px}.tab-btn{padding:8px 16px}.tab-btn.active{border-color:#7be495;color:#7be495;background:#172019}
.vllm-form{display:grid;grid-template-columns:minmax(260px,2fr) repeat(3,minmax(110px,1fr)) minmax(120px,1fr);gap:8px;align-items:end}
.vllm-form label{font-size:11px;color:#aaa}.vllm-form select,.vllm-form input{display:block;width:100%;margin-top:4px;padding:7px}.vllm-check{display:flex!important;align-items:center;gap:7px;padding:7px 4px}.vllm-check input{width:auto!important;margin:0!important}
.vllm-load-wrap{margin-top:10px}.vllm-load-line{display:flex;justify-content:space-between;gap:10px;font-size:11px;color:#bbb}.vllm-load-bar{height:9px;background:#2b2b2b;border:1px solid #383838;border-radius:6px;overflow:hidden;margin-top:5px}.vllm-load-fill{height:100%;width:0%;background:#8a8a8a;transition:width .35s ease}.vllm-load-sub{font-size:10px;color:#888;margin-top:4px}
.vllm-actions,.vllm-bench-buttons{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin-top:10px}.vllm-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin-top:12px}.vllm-cards .worker-stat b{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.vllm-log-wrap{margin-top:10px;background:#101010;border:1px solid #303030;border-radius:8px;padding:8px}.vllm-log-head{display:flex;justify-content:space-between;align-items:center;gap:8px;font-size:10px;color:#888}.vllm-log-head button{padding:4px 8px;font-size:10px}.vllm-log-wrap pre{margin:7px 0 0;max-height:300px;overflow:auto;white-space:pre-wrap;word-break:break-word;font-size:10px;line-height:1.35;color:#cfcfcf}
.vllm-host-strip{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:7px;margin-top:9px}.vllm-mini{background:#171717;border:1px solid #303030;border-radius:8px;padding:7px 9px;font-size:10px;color:#aaa;min-width:0}.vllm-mini b{display:block;margin-top:2px;font-size:17px;line-height:1.15;color:#eee;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.vllm-mini b.ok{color:var(--ok)}.vllm-mini b.warn{color:var(--warn)}.vllm-mini b.bad{color:var(--bad)}.vllm-mini small{display:block;margin-top:2px;color:#888;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.vllm-prompt{width:100%;min-height:120px;resize:vertical;background:#111;color:#eee;border:1px solid #444;border-radius:8px;padding:10px;font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace}.vllm-prompt-actions{display:flex;gap:8px;align-items:end;flex-wrap:wrap;margin-top:8px}.vllm-prompt-actions label{font-size:10px;color:#aaa}.vllm-prompt-actions input{display:block;width:100px;margin-top:3px;padding:6px}.vllm-prompt-output{margin:10px 0 0;max-height:420px;overflow:auto;white-space:pre-wrap;word-break:break-word;background:#101010;border:1px solid #303030;border-radius:8px;padding:10px;font-size:11px;line-height:1.4;color:#ddd}
.vllm-bench-meta{display:flex;gap:7px;flex-wrap:wrap;margin:2px 0 10px}.vllm-bench-meta span{background:#171717;border:1px solid #303030;border-radius:7px;padding:5px 7px;font-size:10px;color:#999}.vllm-bench-meta b{color:#eee;font-weight:700;margin-left:3px}
.vllm-table-wrap{overflow:auto;margin-top:10px}.vllm-table th,.vllm-table td{text-align:left;padding:7px 8px;border-bottom:1px solid #2d2d2d}.vllm-table th{color:#bbb;font-size:11px}.vllm-table td{font-size:12px}
@media(max-width:900px){.vllm-form{grid-template-columns:1fr 1fr}.vllm-host-strip{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:560px){.vllm-host-strip{grid-template-columns:repeat(2,minmax(0,1fr))}}
'''
    dashboard = dashboard.replace("</style>", css + "\n</style>", 1)

    js = r'''
let vllmModelsLoaded=false;
let vllmPrevMetrics=null;
let vllmPrevTs=null;

function showTopTab(which){
 const mon=document.getElementById('monitorTab'),vl=document.getElementById('vllmTab');
 const mb=document.getElementById('tabMonitorBtn'),vb=document.getElementById('tabVllmBtn');
 const isV=which==='vllm';
 mon.style.display=isV?'none':'block';vl.style.display=isV?'block':'none';
 mb.classList.toggle('active',!isV);vb.classList.toggle('active',isV);
 if(isV){loadVllmModels();refreshVllm();}
}

async function loadVllmModels(){
 if(vllmModelsLoaded)return;
 try{
  const r=await fetch('/api/vllm/models');const d=await r.json();
  const el=document.getElementById('vllmModel');
  el.innerHTML=(d.models||[]).map(m=>'<option value="'+m.path.replace(/"/g,'&quot;')+'">'+m.label+'</option>').join('');
  const preferred=(d.models||[]).findIndex(m=>m.label.toLowerCase().includes('qwen3-4b-awq'));
  if(preferred>=0)el.selectedIndex=preferred;
  vllmModelsLoaded=true;
 }catch(e){vllmMsg.textContent='Model list error: '+e;}
}

async function startVllm(){
 await loadVllmModels();
 const model=vllmModel.value;if(!model){vllmMsg.textContent='No vLLM model found';return;}
 const payload={model:model,port:Number(vllmPort.value),max_model_len:Number(vllmCtx.value),gpu_memory_utilization:Number(vllmMem.value),enforce_eager:vllmEager.checked,dtype:'half'};
 vllmMsg.textContent='Starting vLLM…';
 try{
  const r=await fetch('/api/vllm/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  vllmMsg.textContent='Started PID '+d.pid+'; model is loading…';setTimeout(refreshVllm,1200);
 }catch(e){vllmMsg.textContent='Start error: '+e.message;}
}

async function controlVllm(action){
 if(action==='stop'&&!confirm('Stop vLLM server?'))return;
 vllmMsg.textContent=action+' vLLM…';
 try{
  const r=await fetch('/api/vllm/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:action})});
  const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  vllmMsg.textContent='OK: '+action;setTimeout(refreshVllm,700);
 }catch(e){vllmMsg.textContent='Control error: '+e.message;}
}

function renderVllmBench(rows){
 const body=document.getElementById('vllmBenchRows');
 if(!rows||!rows.length){body.innerHTML='<tr><td colspan="5" class="muted">No UI benchmark runs yet</td></tr>';return;}
 body.innerHTML=[...rows].reverse().map(x=>'<tr><td>'+x.concurrency+'</td><td>'+x.total_tokens+'</td><td>'+x.wall_s.toFixed(3)+' s</td><td><b>'+x.aggregate_tok_s.toFixed(2)+' tok/s</b></td><td>'+x.per_request_min_tok_s.toFixed(2)+'–'+x.per_request_max_tok_s.toFixed(2)+' tok/s</td></tr>').join('');
}

async function refreshVllm(){
 try{
  const r=await fetch('/api/vllm/status');const s=await r.json();if(!r.ok)throw new Error(JSON.stringify(s));
  vllmState.textContent=s.ready?'● READY':s.running?'● LOADING':'● STOPPED';
  vllmState.className='status '+(s.ready?'ready':s.running?'loading':'down');
  const lw=document.getElementById('vllmLoadWrap');
  if(s.running&&!s.ready){
    lw.style.display='block';
    let pct=10,txt='Starting vLLM…',sub='Process started; waiting for model initialization…';
    const vramText=(document.getElementById('vllmGpuVram')||{}).textContent||'';
    const vm=parseFloat(vramText);
    if(Number.isFinite(vm)){
      if(vm>0.2){pct=30;txt='Loading weights…';sub='Model weights are being mapped into GPU memory.';}
      if(vm>2.0){pct=55;txt='Loading model…';sub='Weights loaded partially; preparing runtime and kernels.';}
      if(vm>5.0){pct=78;txt='Initializing engine…';sub='Model is mostly resident; preparing KV cache / attention kernels.';}
      if(vm>6.0){pct=90;txt='Warming up…';sub='Final warmup / compilation before the API becomes ready.';}
    }
    vllmLoadFill.style.width=pct+'%';vllmLoadPct.textContent=pct+'%';vllmLoadText.textContent=txt;vllmLoadSub.textContent=sub;
  }else if(s.ready){
    lw.style.display='none';
    vllmLoadFill.style.width='100%';
    vllmLoadPct.textContent='100%';
    vllmLoadText.textContent='Ready';
    vllmLoadSub.textContent='Model loaded and API ready.';
  }else{
    lw.style.display='none';vllmLoadFill.style.width='0%';vllmLoadPct.textContent='0%';
  }
  vllmPid.textContent=s.pid??'—';vllmLiveModel.textContent=s.model||'—';vllmUptime.textContent=s.running?fmtUptime(s.uptime_s):'—';
  const m=s.metrics||{};vllmRunning.textContent=m.running==null?'—':Math.round(m.running);vllmWaiting.textContent=m.waiting==null?'—':Math.round(m.waiting);
  vllmKv.textContent=m.kv_cache_usage==null?'—':(m.kv_cache_usage*100).toFixed(1)+'%';
  const now=performance.now()/1000;
  if(vllmPrevMetrics&&vllmPrevTs&&m.generation_tokens_total!=null){
   const dt=Math.max(.1,now-vllmPrevTs);
   const dg=m.generation_tokens_total-vllmPrevMetrics.generation_tokens_total;
   const dp=m.prompt_tokens_total-vllmPrevMetrics.prompt_tokens_total;
   vllmDecode.textContent=(dg/dt).toFixed(1)+' tok/s';vllmPrompt.textContent=(dp/dt).toFixed(1)+' tok/s';
  }else{vllmDecode.textContent='—';vllmPrompt.textContent='—';}
  if(m.generation_tokens_total!=null){vllmPrevMetrics={generation_tokens_total:m.generation_tokens_total,prompt_tokens_total:m.prompt_tokens_total||0};vllmPrevTs=now;}
  if(s.port)vllmPort.value=s.port;if(s.max_model_len)vllmCtx.value=s.max_model_len;if(s.gpu_memory_utilization)vllmMem.value=s.gpu_memory_utilization.toFixed(2);vllmEager.checked=!!s.enforce_eager;
  renderVllmBench(s.benchmarks||[]);
  benchModel.textContent=(s.model||'—').split('/').pop();
  benchQuant.textContent=s.quantization||'—';
  benchCtx.textContent=s.max_model_len?Number(s.max_model_len).toLocaleString():'—';
  benchMode.textContent=s.enforce_eager?'eager':'compiled/graphs';
  benchVllm.textContent=s.vllm_version||'—';
  try{
   const hr=await fetch('/api/stats');const h=await hr.json();
   if(hr.ok){
    vllmCpu.textContent=h.cpu.usage_pct.toFixed(1)+'%';
    vllmCpuSub.textContent='load '+h.cpu.load_1m.toFixed(2)+' • '+(h.cpu.temperature_c==null?'?':h.cpu.temperature_c.toFixed(0)+'°C');
    vllmRam.textContent=h.memory.usage_pct.toFixed(1)+'%';
    vllmRamSub.textContent=(h.memory.used_bytes/1073741824).toFixed(2)+' / '+(h.memory.total_bytes/1073741824).toFixed(2)+' GiB';
    const g=(h.gpu.devices||[])[0];
    if(g){
     vllmGpuLoad.textContent=(g.utilization_pct??'?')+'%';
     vllmGpuName.textContent=g.name||'GPU #0';
     vllmGpuPower.textContent=(g.power_w??'?')+' W';
     vllmGpuPowerSub.textContent='limit '+(g.power_limit_w??'?')+' W';
     vllmGpuVram.textContent=((g.memory_used_mib||0)/1024).toFixed(2)+' GiB';
     vllmGpuVramSub.textContent='of '+((g.memory_total_mib||0)/1024).toFixed(2)+' GiB';
     const gt=g.temperature_c;
     vllmGpuTemp.textContent=(gt??'?')+'°C';
     vllmGpuTemp.className=gt==null?'':cls(gt);
     vllmGpuFan.textContent=g.fan_pct==null?'fan N/A':'fan '+g.fan_pct+'%';
     benchPl.textContent=(g.power_limit_w??'?')+' W';
     benchGpu.textContent=g.name||'GPU #0';
    }
   }
  }catch(_e){}
 }catch(e){vllmState.textContent='● ERROR';vllmState.className='status error';vllmMsg.textContent='Status error: '+e.message;}
}

async function runVllmPrompt(){
 const prompt=document.getElementById('vllmPromptInput').value.trim();
 if(!prompt){vllmPromptMsg.textContent='Enter a prompt first';return;}
 const payload={prompt:prompt,max_tokens:Number(vllmPromptTokens.value),temperature:Number(vllmPromptTemp.value)};
 vllmPromptMsg.textContent='Running…';vllmPromptOutput.textContent='Generating…';
 try{
  const r=await fetch('/api/vllm/prompt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  vllmPromptOutput.textContent=d.content||'(empty response)';
  const u=d.usage||{};
  vllmPromptMsg.textContent=(u.prompt_tokens??'?')+' prompt + '+(u.completion_tokens??'?')+' output • '+d.elapsed_s.toFixed(3)+' s • '+d.output_tok_s.toFixed(2)+' tok/s';
 }catch(e){vllmPromptOutput.textContent='ERROR: '+e.message;vllmPromptMsg.textContent='Request failed';}
}
function clearVllmPrompt(){vllmPromptOutput.textContent='Response will appear here.';vllmPromptMsg.textContent='';}

function downloadVllmReport(){
 window.location.href='/api/vllm/report';
}

async function refreshVllmLogs(){
 try{
  const r=await fetch('/api/vllm/logs');const d=await r.json();
  vllmLogPath.textContent=d.path||'vLLM log';
  vllmLogText.textContent=d.log||'(empty)';
  vllmLogText.scrollTop=vllmLogText.scrollHeight;
 }catch(e){vllmLogText.textContent='Log error: '+e.message;}
}

async function toggleVllmLogs(){
 const w=document.getElementById('vllmLogWrap');
 if(w.style.display==='none'){w.style.display='block';await refreshVllmLogs();}else{w.style.display='none';}
}

async function runVllmBench(concurrency){
 vllmBenchMsg.textContent='Running '+concurrency+' × 512…';
 try{
  const r=await fetch('/api/vllm/benchmark',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({concurrency:concurrency,max_tokens:512})});
  const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  vllmBenchMsg.textContent=concurrency+' × 512 = '+d.result.aggregate_tok_s.toFixed(2)+' tok/s';
  await refreshVllm();
 }catch(e){vllmBenchMsg.textContent='Benchmark error: '+e.message;}
}
setInterval(()=>{if(document.getElementById('vllmTab')&&document.getElementById('vllmTab').style.display!=='none')refreshVllm();},2000);
'''
    dashboard = dashboard.replace(
        "refresh();setInterval(refresh,2000);",
        js + "\nrefresh();setInterval(refresh,2000);",
        1,
    )

    base.DASHBOARD = dashboard
    dynamic.base.DASHBOARD = dashboard
    dynamic.app = base.app

#!/usr/bin/env python3
"""vLLM control/status tab for the AI6 Host Monitor."""

import concurrent.futures
import hashlib
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
from fastapi.responses import PlainTextResponse, HTMLResponse

import ai6_monitor_dynamic as dynamic

base = dynamic.base

_VLLM_ENV = Path(os.environ.get("AI6_VLLM_ENV", str(Path.home() / "vllm-env")))
_VLLM_BIN = Path(os.environ.get("AI6_VLLM_BIN", str(_VLLM_ENV / "bin/vllm")))
_VLLM_MODELS_DIR = Path(os.environ.get("AI6_VLLM_MODELS_DIR", str(Path.home() / "models/vllm")))
_VLLM_LOG_DIR = Path(os.environ.get("AI6_VLLM_LOG_DIR", str(Path.home() / ".local/state/ai6-monitor/vllm")))
_VLLM_DEFAULT_PORT = int(os.environ.get("AI6_VLLM_PORT", "8012"))
_STATE_DIR = Path(os.environ.get("AI6_MONITOR_STATE_DIR", str(Path.home() / ".local/state/ai6-monitor")))
_BENCH_STORE = Path(os.environ.get("AI6_VLLM_BENCH_STORE", str(_STATE_DIR / "vllm-benchmarks.json")))
_REPO_ROOT = Path(__file__).resolve().parents[1]
_BENCH_RESULTS = []
_LAST_PROMPT_RESULT = None
_LOAD_TIMELINE = {}

try:
    if _BENCH_STORE.is_file():
        data = json.loads(_BENCH_STORE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            _BENCH_RESULTS.extend(x for x in data if isinstance(x, dict))
except Exception:
    pass

try:
    _VLLM_VERSION = package_version("vllm")
except PackageNotFoundError:
    try:
        p = subprocess.run(
            [str(_VLLM_BIN), "--version"],
            capture_output=True, text=True, timeout=3.0,
        )
        m = re.search(r"(\d+\.\d+\.\d+(?:[-+._A-Za-z0-9]*)?)", (p.stdout or "") + " " + (p.stderr or ""))
        _VLLM_VERSION = m.group(1) if m else "unknown"
    except Exception:
        _VLLM_VERSION = "unknown"


class VllmStartCommand(BaseModel):
    model: str
    port: int = Field(default=_VLLM_DEFAULT_PORT, ge=1024, le=65535)
    max_model_len: int = Field(default=4096, ge=256, le=262144)
    gpu_memory_utilization: float = Field(default=0.90, ge=0.10, le=0.99)
    enforce_eager: bool = False
    dtype: str = "half"
    gpu_ids: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=16)


class VllmControlCommand(BaseModel):
    action: str


class VllmBenchmarkCommand(BaseModel):
    concurrency: int = Field(ge=1, le=128)
    max_tokens: int = Field(default=512, ge=16, le=8192)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    prompt_profile: str = Field(default="short")
    enable_thinking: bool = False


class VllmPromptCommand(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)
    max_tokens: int = Field(default=512, ge=16, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class GpuPowerLimitCommand(BaseModel):
    watts: float = Field(ge=1, le=1000)
    gpu: int = Field(default=0, ge=0, le=31)


class BenchmarkMetaCommand(BaseModel):
    run_id: str = Field(min_length=1, max_length=128)
    selected: bool | None = None
    comment: str | None = Field(default=None, max_length=500)


class BenchmarkDeleteCommand(BaseModel):
    run_ids: list[str] = Field(min_length=1, max_length=500)


def _save_bench_results():
    _BENCH_STORE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _BENCH_STORE.with_suffix(".tmp")
    tmp.write_text(json.dumps(_BENCH_RESULTS[-500:], ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_BENCH_STORE)


def _find_bench(run_id):
    return next((x for x in _BENCH_RESULTS if x.get("run_id") == run_id), None)


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

    gpu_ids = [0]
    try:
        visible = (proc.environ() or {}).get("CUDA_VISIBLE_DEVICES", "")
        parsed = [int(x.strip()) for x in visible.split(",") if x.strip().isdigit()]
        if parsed:
            gpu_ids = parsed
    except Exception:
        pass

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
        "gpu_ids": gpu_ids,
        "tensor_parallel_size": len(gpu_ids),
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

    gpu_stats = base.gpu_stats() or []
    gpu_ids = sorted(dict.fromkeys(int(x) for x in cmd.gpu_ids))
    if not gpu_ids:
        raise base.HTTPException(status_code=400, detail="select at least one GPU")
    invalid = [i for i in gpu_ids if i < 0 or i >= len(gpu_stats)]
    if invalid:
        raise base.HTTPException(status_code=400, detail=f"invalid GPU indices: {invalid}")

    args = [
        str(_VLLM_BIN), "serve", str(model),
        "--host", "0.0.0.0",
        "--port", str(cmd.port),
        "--dtype", cmd.dtype,
        "--gpu-memory-utilization", f"{cmd.gpu_memory_utilization:.3f}",
        "--max-model-len", str(cmd.max_model_len),
        "--tensor-parallel-size", str(len(gpu_ids)),
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
            env={**os.environ.copy(), "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in gpu_ids)},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    except Exception as e:
        log.close()
        raise base.HTTPException(status_code=409, detail=f"failed to start vLLM: {e}")
    log.close()
    return {"pid": proc.pid, "port": cmd.port, "log": str(log_path), "model": str(model), "gpu_ids": gpu_ids, "tensor_parallel_size": len(gpu_ids)}


def _gpu_power_limits(gpu=0):
    try:
        p = subprocess.run(
            [
                "nvidia-smi", "-i", str(gpu),
                "--query-gpu=power.limit,power.default_limit,power.min_limit,power.max_limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=3.0,
        )
        if p.returncode != 0:
            return {}
        parts = [x.strip() for x in (p.stdout or "").strip().split(",")]
        if len(parts) < 4:
            return {}
        current, default, minimum, maximum = [float(x) for x in parts[:4]]
        return {
            "current_w": current,
            "default_w": default,
            "min_w": minimum,
            "max_w": maximum,
        }
    except Exception:
        return {}


@base.app.get("/api/gpu/power-limit")
def api_gpu_power_limit(gpu: int = 0):
    info = _gpu_power_limits(gpu)
    if not info:
        raise base.HTTPException(status_code=503, detail="failed to read GPU power limits")
    return {"ok": True, "gpu": gpu, **info}


@base.app.post("/api/gpu/power-limit")
def api_set_gpu_power_limit(cmd: GpuPowerLimitCommand):
    info = _gpu_power_limits(cmd.gpu)
    if not info:
        raise base.HTTPException(status_code=503, detail="failed to read GPU power limits")
    if not (info["min_w"] <= cmd.watts <= info["max_w"]):
        raise base.HTTPException(
            status_code=400,
            detail=f"power limit must be {info['min_w']:.0f}..{info['max_w']:.0f} W",
        )

    helper = "/usr/local/sbin/ai6-gpuctl"
    attempts = [
        ["nvidia-smi", "-i", str(cmd.gpu), "-pl", f"{cmd.watts:.0f}"],
        ["sudo", "-n", helper, "power-limit", str(cmd.gpu), f"{cmd.watts:.0f}"],
    ]
    last_error = ""
    for args in attempts:
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=5.0)
            if p.returncode == 0:
                new_info = _gpu_power_limits(cmd.gpu)
                return {"ok": True, "gpu": cmd.gpu, **new_info}
            last_error = (p.stderr or p.stdout or "").strip()
        except Exception as e:
            last_error = str(e)

    raise base.HTTPException(
        status_code=403,
        detail=(
            "failed to set power limit: " + (last_error or "permission denied") +
            ". Install the restricted AI6 GPU helper with monitor/install-gpu-control.sh."
        ),
    )


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


@base.app.get("/api/vllm/benchmarks")
def api_vllm_benchmarks():
    return {
        "ok": True,
        "store": str(_BENCH_STORE),
        "count": len(_BENCH_RESULTS),
        "benchmarks": list(_BENCH_RESULTS),
    }


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
        gpu_ids=status.get("gpu_ids") or [0],
    )
    _stop_vllm()
    time.sleep(0.5)
    result = _start_vllm(start_cmd)
    return {"ok": True, **result}


def _bench_one(port, model, prompt, max_tokens, temperature=0.0, enable_thinking=False):
    """Run one streaming request and capture TTFT + exact token usage."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at = None
    prompt_tokens = 0
    completion_tokens = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if not body or body == "[DONE]":
                continue
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                continue
            choices = data.get("choices") or []
            if choices:
                delta = (choices[0].get("delta") or {}).get("content")
                if delta and first_token_at is None:
                    first_token_at = time.perf_counter()
            usage = data.get("usage") or {}
            if usage:
                prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens or 0)
                completion_tokens = int(usage.get("completion_tokens") or completion_tokens or 0)

    finished = time.perf_counter()
    elapsed = finished - started
    ttft = (first_token_at - started) if first_token_at is not None else elapsed
    decode_s = max(0.0, elapsed - ttft)
    decode_tok_s = completion_tokens / decode_s if decode_s > 0 and completion_tokens else 0.0
    e2e_tok_s = completion_tokens / elapsed if elapsed > 0 and completion_tokens else 0.0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed_s": elapsed,
        "ttft_s": ttft,
        "decode_s": decode_s,
        "decode_tok_s": decode_tok_s,
        "e2e_tok_s": e2e_tok_s,
    }


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


def _system_cuda_info():
    driver = None
    cuda = None
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3.0,
        )
        if p.returncode == 0:
            vals = [x.strip() for x in (p.stdout or "").splitlines() if x.strip()]
            if vals:
                driver = vals[0]
    except Exception:
        pass
    try:
        p = subprocess.run(
            [str(_VLLM_ENV / "bin/python"), "-c",
             "import torch; print(torch.__version__); print(torch.version.cuda or '')"],
            capture_output=True, text=True, timeout=5.0,
        )
        if p.returncode == 0:
            lines = [x.strip() for x in (p.stdout or "").splitlines()]
            torch_ver = lines[0] if lines else ""
            cuda_ver = lines[1] if len(lines) > 1 else ""
            cuda = {"torch": torch_ver, "cuda_runtime": cuda_ver}
    except Exception:
        pass
    return {"driver_version": driver, "cuda": cuda or {}}


def _pcie_info_for_gpu(index):
    info = {
        "index": index,
        "bus_id": None,
        "gen_current": None,
        "width_current": None,
        "gen_max": None,
        "width_max": None,
    }
    try:
        p = subprocess.run(
            [
                "nvidia-smi", "-i", str(index),
                "--query-gpu=pci.bus_id,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=3.0,
        )
        if p.returncode == 0 and p.stdout.strip():
            parts = [x.strip() for x in p.stdout.strip().split(",")]
            if len(parts) >= 5:
                info.update({
                    "bus_id": parts[0],
                    "gen_current": int(float(parts[1])) if parts[1] else None,
                    "width_current": int(float(parts[2])) if parts[2] else None,
                    "gen_max": int(float(parts[3])) if parts[3] else None,
                    "width_max": int(float(parts[4])) if parts[4] else None,
                })
    except Exception:
        pass
    return info


def _gpu_config_label(x):
    devices = x.get("gpu_devices") or []
    if devices:
        return " + ".join(
            f"GPU{g.get('index','?')} {g.get('name') or 'GPU'}"
            for g in devices
        )
    count = x.get("gpu_count") or 1
    ids = x.get("gpu_ids") or [0]
    return f"{count} GPU(s): " + ", ".join(f"GPU{i}" for i in ids)


def _read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return ""


def _host_platform_info():
    """Best-effort host metadata for benchmark reproducibility."""
    cpu_model = ""
    try:
        for line in _read_text("/proc/cpuinfo").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass

    logical = psutil.cpu_count(logical=True)
    physical = psutil.cpu_count(logical=False)
    sockets = None
    try:
        p = subprocess.run(["lscpu", "-J"], capture_output=True, text=True, timeout=2.0)
        if p.returncode == 0:
            data = json.loads(p.stdout)
            kv = {str(x.get("field","")).rstrip(":"): str(x.get("data","")) for x in data.get("lscpu", [])}
            sockets = int(kv.get("Socket(s)", "0") or 0) or None
            cpu_model = cpu_model or kv.get("Model name", "")
    except Exception:
        pass

    vendor = _read_text("/sys/devices/virtual/dmi/id/sys_vendor")
    product = _read_text("/sys/devices/virtual/dmi/id/product_name")
    board_vendor = _read_text("/sys/devices/virtual/dmi/id/board_vendor")
    board_name = _read_text("/sys/devices/virtual/dmi/id/board_name")

    os_name = ""
    try:
        vals = {}
        for line in _read_text("/etc/os-release").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                vals[k] = v.strip().strip('"')
        os_name = vals.get("PRETTY_NAME") or vals.get("NAME") or ""
    except Exception:
        pass

    ram_total_gib = round(psutil.virtual_memory().total / (1024**3), 2)
    ram_desc = ""
    ram_speed = ""
    # lshw sometimes exposes type/clock without root; keep best-effort only.
    try:
        p = subprocess.run(
            ["lshw", "-class", "memory"],
            capture_output=True, text=True, timeout=3.0,
        )
        txt = (p.stdout or "") + "\n" + (p.stderr or "")
        types = sorted(set(re.findall(r"\b(?:DDR[2-5]|LPDDR[3-5]|SDRAM)\b", txt, re.I)))
        clocks = sorted(set(re.findall(r"clock:\s*([^\n]+)", txt, re.I)))
        if types:
            ram_desc = ", ".join(types)
        if clocks:
            ram_speed = ", ".join(clocks[:4])
    except Exception:
        pass

    return {
        "cpu_model": cpu_model or None,
        "cpu_sockets": sockets,
        "cpu_physical_cores": physical,
        "cpu_logical_threads": logical,
        "ram_total_gib": ram_total_gib,
        "ram_type": ram_desc or None,
        "ram_speed": ram_speed or None,
        "system_vendor": vendor or None,
        "product_name": product or None,
        "board_vendor": board_vendor or None,
        "board_name": board_name or None,
        "os": os_name or None,
        "kernel": os.uname().release if hasattr(os, "uname") else None,
    }


def _host_label(host):
    host = host or {}
    bits = [host.get("system_vendor"), host.get("product_name")]
    return " ".join(str(x) for x in bits if x) or "-"


def _host_for_report(x):
    host = x.get("host") if isinstance(x.get("host"), dict) else None
    if host:
        return host, "recorded with run"
    # Legacy rows predate host metadata. Current snapshot is shown explicitly,
    # never silently represented as historical metadata.
    return _host_platform_info(), "current host snapshot (legacy run; not stored at run time)"


def _single_benchmark_report(x):
    eff = None
    if x.get("gpu_power_avg_w") and x.get("aggregate_tok_s"):
        eff = x["aggregate_tok_s"] / x["gpu_power_avg_w"]
    lines = [
        "# AI6 vLLM benchmark run",
        "",
        f"Run ID: {x.get('run_id', '-')}",
        f"Timestamp: {x.get('timestamp', '-')}",
        f"Comment: {x.get('comment') or '-'}",
        "",
        "## Model / runtime",
        f"- Model: {x.get('model') or '-'}",
        f"- Quantization: {x.get('quantization') or '-'}",
        f"- Context: {x.get('context') or '-'}",
        f"- Mode: {x.get('mode') or '-'}",
        f"- vLLM: {x.get('vllm_version') or '-'}",
        f"- GPU count: {x.get('gpu_count') or 1}",
        f"- GPU configuration: {_gpu_config_label(x)}",
        f"- NVIDIA driver: {x.get('driver_version') or '-'}",
        f"- Torch: {(x.get('cuda') or {}).get('torch') or '-'}",
        f"- CUDA runtime: {(x.get('cuda') or {}).get('cuda_runtime') or '-'}",
        "",
        "## Host platform",
    ]
    host, host_source = _host_for_report(x)
    lines += [
        f"- Metadata source: {host_source}",
        f"- Platform: {_host_label(host)}",
        f"- Board: {' '.join(str(v) for v in [host.get('board_vendor'), host.get('board_name')] if v) or '-'}",
        f"- CPU: {host.get('cpu_model') or '-'}",
        f"- CPU sockets: {host.get('cpu_sockets') or '-'}",
        f"- CPU cores / threads: {host.get('cpu_physical_cores') or '-'} / {host.get('cpu_logical_threads') or '-'}",
        f"- RAM: {host.get('ram_total_gib') or '-'} GiB"
           + (f" • {host.get('ram_type')}" if host.get('ram_type') else "")
           + (f" • {host.get('ram_speed')}" if host.get('ram_speed') else ""),
        f"- OS: {host.get('os') or '-'}",
        f"- Kernel: {host.get('kernel') or '-'}",
        "",
        "## Request shape",
        f"- Concurrent requests: {x.get('concurrency', '-')}",
        f"- Prompt profile: {x.get('prompt_profile') or '-'}",
        f"- Prompt profile ID: {x.get('prompt_profile_id') or '-'}",
        f"- Prompt set SHA256: {x.get('prompt_set_hash') or '-'}",
        f"- First prompt SHA256: {x.get('prompt_first_hash') or '-'}",
        f"- Temperature: {x.get('temperature', '-')}",
        f"- Thinking: {'ON' if x.get('enable_thinking') else 'OFF'}",
        f"- Prompt tokens total: {x.get('prompt_tokens', '-')}",
        f"- Output tokens/request: {x.get('max_tokens', '-')}",
        f"- Output tokens total: {x.get('output_tokens', '-')}",
        f"- Total tokens: {x.get('total_tokens', '-')}",
        "",
        "## Latency / throughput",
        f"- TTFT average: {x.get('ttft_avg_s', 0):.3f} s",
        f"- TTFT max: {x.get('ttft_max_s', 0):.3f} s",
        f"- Approx. prompt throughput: {x.get('prompt_tok_s_approx', 0):.2f} tok/s",
        f"- Wall time: {x.get('wall_s', 0):.3f} s",
        f"- Aggregate output throughput: {x.get('aggregate_tok_s', 0):.2f} tok/s",
        f"- Per-request output throughput: {x.get('per_request_min_tok_s', 0):.2f}-{x.get('per_request_max_tok_s', 0):.2f} tok/s",
        f"- Average decode speed/request: {x.get('decode_avg_tok_s', 0):.2f} tok/s",
        "",
        "## PCIe",
    ]
    pcie_rows = x.get("pcie") or []
    if pcie_rows:
        for p in pcie_rows:
            lines.append(
                f"- GPU{p.get('index','?')} bus {p.get('bus_id') or '-'}: "
                f"Gen{p.get('gen_current') or '?'} x{p.get('width_current') or '?'} current; "
                f"max Gen{p.get('gen_max') or '?'} x{p.get('width_max') or '?'}"
            )
    else:
        lines.append("- No PCIe data recorded.")
    lines += [
        "",
        "## Telemetry",
        f"- Average GPU load: {x.get('gpu_load_avg_pct') if x.get('gpu_load_avg_pct') is not None else '-'} %",
        f"- Peak GPU load: {x.get('gpu_load_peak_pct') if x.get('gpu_load_peak_pct') is not None else '-'} %",
        f"- Average GPU power: {x.get('gpu_power_avg_w') if x.get('gpu_power_avg_w') is not None else '-'} W",
        f"- Peak GPU power: {x.get('gpu_power_peak_w') if x.get('gpu_power_peak_w') is not None else '-'} W",
        f"- Power limit: {x.get('gpu_power_limit_w') if x.get('gpu_power_limit_w') is not None else '-'} W",
        f"- Peak VRAM: {round((x.get('gpu_vram_peak_mib') or 0)/1024, 2) if x.get('gpu_vram_peak_mib') is not None else '-'} GiB",
        f"- Peak temperature: {x.get('gpu_temp_peak_c') if x.get('gpu_temp_peak_c') is not None else '-'} C",
        f"- Peak fan: {x.get('gpu_fan_peak_pct') if x.get('gpu_fan_peak_pct') is not None else '-'} %",
        f"- Average CPU: {x.get('cpu_avg_pct') if x.get('cpu_avg_pct') is not None else '-'} %",
        "- CPU contribution to token generation: not directly separable from utilization telemetry; "
        "CPU load includes scheduling, tokenization, API/runtime work and does not equal a percentage of generated tokens.",
        f"- Average RAM: {x.get('ram_avg_pct') if x.get('ram_avg_pct') is not None else '-'} %",
        f"- Efficiency: {eff:.3f} aggregate tok/s/W" if eff is not None else "- Efficiency: -",
        f"- Telemetry samples: {x.get('sample_count', 0)}",
        "",
        "## Per-GPU telemetry",
    ]
    for g in (x.get("gpu_devices") or []):
        lines += [
            f"- GPU{g.get('index','?')} {g.get('name') or 'GPU'}: "
            f"load {g.get('load_avg_pct','-')}/{g.get('load_peak_pct','-')} %, "
            f"power {g.get('power_avg_w','-')}/{g.get('power_peak_w','-')} W, "
            f"PL {g.get('power_limit_w','-')} W, "
            f"VRAM peak {round((g.get('vram_peak_mib') or 0)/1024,2)} GiB, "
            f"temp {g.get('temp_peak_c','-')} C",
        ]
    lines += [
        "",
        "*Prompt tok/s is approximate: total prompt tokens divided by the slowest TTFT in the batch; TTFT includes queueing and first-token overhead.*",
        "",
    ]
    return "\n".join(lines)


def _combined_benchmark_report(rows):
    rows = list(rows)
    lines = [
        "# AI6 vLLM selected benchmark report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"Selected runs: {len(rows)}",
        "",
        "## Comparison",
        "",
        "| Date | Model / quant | GPUs | PCIe | Profile | Temp | Thinking | Conc | PL total | Prompt tok | Out/req | TTFT avg/max | Wall s | Aggregate tok/s | Per-request tok/s | Avg/peak power total | Peak temp | Comment |",
        "|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for x in rows:
        date = (x.get("timestamp") or "-").replace("T", " ")[:19]
        model_quant = f"{x.get('model') or '-'} / {x.get('quantization') or '-'}"
        pl = x.get("gpu_power_limit_w")
        avg_p = x.get("gpu_power_avg_w")
        peak_p = x.get("gpu_power_peak_w")
        temp = x.get("gpu_temp_peak_c")
        comment = (x.get("comment") or "-").replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {date} | {model_quant} | {_gpu_config_label(x)} | "
            f"{'; '.join('GPU'+str(p.get('index','?'))+':G'+str(p.get('gen_current') or '?')+'x'+str(p.get('width_current') or '?') for p in (x.get('pcie') or [])) or '-'} | "
            f"{x.get('prompt_profile') or '-'} | {x.get('temperature','-')} | {'ON' if x.get('enable_thinking') else 'OFF'} | "
            f"{x.get('concurrency','-')} | "
            f"{f'{pl:.0f} W' if isinstance(pl,(int,float)) else '-'} | "
            f"{x.get('prompt_tokens','-')} | {x.get('max_tokens','-')} | "
            f"{x.get('ttft_avg_s',0):.3f}/{x.get('ttft_max_s',0):.3f} s | "
            f"{x.get('wall_s',0):.3f} | {x.get('aggregate_tok_s',0):.2f} | "
            f"{x.get('per_request_min_tok_s',0):.2f}-{x.get('per_request_max_tok_s',0):.2f} | "
            f"{f'{avg_p:.1f}/{peak_p:.1f} W' if isinstance(avg_p,(int,float)) and isinstance(peak_p,(int,float)) else '-'} | "
            f"{f'{temp:.0f} C' if isinstance(temp,(int,float)) else '-'} | {comment} |"
        )

    if rows:
        best_agg = max(rows, key=lambda x: float(x.get("aggregate_tok_s") or 0))
        eff_rows = [
            (x, float(x.get("aggregate_tok_s") or 0) / float(x.get("gpu_power_avg_w") or 1))
            for x in rows if x.get("gpu_power_avg_w")
        ]
        lines += [
            "",
            "## Highlights",
            f"- Highest aggregate throughput: {best_agg.get('aggregate_tok_s',0):.2f} tok/s — {best_agg.get('model') or '-'}, concurrency {best_agg.get('concurrency','-')}, PL {best_agg.get('gpu_power_limit_w','-')} W.",
        ]
        if eff_rows:
            best_eff, eff = max(eff_rows, key=lambda pair: pair[1])
            lines.append(
                f"- Highest measured efficiency: {eff:.3f} aggregate tok/s/W — {best_eff.get('model') or '-'}, "
                f"concurrency {best_eff.get('concurrency','-')}, PL {best_eff.get('gpu_power_limit_w','-')} W."
            )

    lines += ["", "## Individual runs", ""]
    for i, x in enumerate(rows, 1):
        lines += [
            f"### Run {i}: {x.get('model') or '-'} / {x.get('quantization') or '-'} — {x.get('concurrency','-')} × {x.get('max_tokens','-')}",
            "",
            _single_benchmark_report(x),
            "",
            "---",
            "",
        ]
    return "\n".join(lines)


def _combined_benchmark_report_html(rows):
    import html
    rows = list(rows)

    def h(v):
        return html.escape(str(v if v is not None else "-"))

    def fmt(v, digits=2):
        return f"{float(v):.{digits}f}" if isinstance(v, (int, float)) else "-"

    body_rows = []
    for i, x in enumerate(rows, 1):
        pl = x.get("gpu_power_limit_w")
        avg_p = x.get("gpu_power_avg_w")
        peak_p = x.get("gpu_power_peak_w")
        temp = x.get("gpu_temp_peak_c")
        eff = None
        if avg_p and x.get("aggregate_tok_s"):
            eff = float(x["aggregate_tok_s"]) / float(avg_p)
        body_rows.append(f"""
        <tr>
          <td>{i}</td>
          <td class="nowrap">{h((x.get("timestamp") or "-").replace("T"," ")[:19])}</td>
          <td><b>{h(x.get("model") or "-")}</b><div class="sub">{h(x.get("quantization") or "-")}</div></td>
          <td>{h(_gpu_config_label(x))}</td>
          <td>{h("; ".join("GPU"+str(p.get("index","?"))+": G"+str(p.get("gen_current") or "?")+" x"+str(p.get("width_current") or "?") for p in (x.get("pcie") or [])) or "-")}</td>
          <td><span class="profile {h((x.get("prompt_profile") or "").lower())}">{h((x.get("prompt_profile") or "-")[:1].upper())}</span><span class="profile-name">{h(x.get("prompt_profile") or "-")}</span></td>
          <td>{h(x.get("temperature") if x.get("temperature") is not None else "-")}</td>
          <td><b>{'ON' if x.get("enable_thinking") else 'OFF'}</b></td>
          <td>{h(x.get("concurrency","-"))}</td>
          <td>{fmt(pl,0)} W</td>
          <td>{h(x.get("prompt_tokens","-"))}</td>
          <td>{h(x.get("max_tokens","-"))}</td>
          <td>{fmt(x.get("ttft_avg_s"),3)} / {fmt(x.get("ttft_max_s"),3)} s</td>
          <td>{fmt(x.get("wall_s"),3)} s</td>
          <td class="hot">{fmt(x.get("aggregate_tok_s"),2)} tok/s</td>
          <td>{fmt(x.get("per_request_min_tok_s"),2)}–{fmt(x.get("per_request_max_tok_s"),2)}</td>
          <td>{fmt(avg_p,1)} / {fmt(peak_p,1)} W</td>
          <td>{fmt(temp,0)}°C</td>
          <td>{fmt(eff,3) if eff is not None else "-"}</td>
          <td class="comment">{h(x.get("comment") or "-")}</td>
        </tr>""")

    highlights = ""
    if rows:
        best_agg = max(rows, key=lambda x: float(x.get("aggregate_tok_s") or 0))
        eff_rows = [
            (x, float(x.get("aggregate_tok_s") or 0) / float(x.get("gpu_power_avg_w") or 1))
            for x in rows if x.get("gpu_power_avg_w")
        ]
        best_eff_html = ""
        if eff_rows:
            best_eff, eff = max(eff_rows, key=lambda pair: pair[1])
            best_eff_html = (
                f'<div class="card"><span>Best efficiency</span><b>{eff:.3f} tok/s/W</b>'
                f'<small>{h(best_eff.get("model") or "-")} • conc {h(best_eff.get("concurrency","-"))} • PL {fmt(best_eff.get("gpu_power_limit_w"),0)} W</small></div>'
            )
        highlights = (
            f'<div class="cards">'
            f'<div class="card"><span>Selected runs</span><b>{len(rows)}</b><small>combined report</small></div>'
            f'<div class="card"><span>Best aggregate</span><b>{fmt(best_agg.get("aggregate_tok_s"),2)} tok/s</b>'
            f'<small>{h(best_agg.get("model") or "-")} • conc {h(best_agg.get("concurrency","-"))} • PL {fmt(best_agg.get("gpu_power_limit_w"),0)} W</small></div>'
            f'{best_eff_html}'
            f'</div>'
        )

    detail_sections = []
    for i, x in enumerate(rows, 1):
        host, host_source = _host_for_report(x)
        avg_p = x.get("gpu_power_avg_w")
        eff = (float(x.get("aggregate_tok_s") or 0) / float(avg_p)) if avg_p else None
        detail_sections.append(f"""
        <details>
          <summary>Run {i}: {h(x.get("model") or "-")} / {h(x.get("quantization") or "-")} — {h(x.get("concurrency","-"))} × {h(x.get("max_tokens","-"))}</summary>
          <div class="detail-grid">
            <div><span>Run ID</span><b>{h(x.get("run_id") or "-")}</b></div>
            <div><span>GPUs</span><b>{h(_gpu_config_label(x))}</b></div>
            <div><span>Driver</span><b>{h(x.get("driver_version") or "-")}</b></div>
            <div><span>CUDA runtime</span><b>{h((x.get("cuda") or {}).get("cuda_runtime") or "-")}</b></div>
            <div><span>Torch</span><b>{h((x.get("cuda") or {}).get("torch") or "-")}</b></div>
            <div><span>PCIe</span><b>{h("; ".join("GPU"+str(p.get("index","?"))+": G"+str(p.get("gen_current") or "?")+" x"+str(p.get("width_current") or "?") for p in (x.get("pcie") or [])) or "-")}</b></div>
            <div><span>Context</span><b>{h(x.get("context") or "-")}</b></div>
            <div><span>Prompt profile</span><b>{h(x.get("prompt_profile") or "-")}</b></div>
            <div><span>Prompt ID</span><b>{h(x.get("prompt_profile_id") or "-")}</b></div>
            <div><span>Prompt hash</span><b title="{h(x.get("prompt_set_hash") or "-")}">{h((x.get("prompt_set_hash") or "-")[:12])}</b></div>
            <div><span>Temperature</span><b>{h(x.get("temperature") if x.get("temperature") is not None else "-")}</b></div>
            <div><span>Thinking</span><b>{'ON' if x.get("enable_thinking") else 'OFF'}</b></div>
            <div><span>Mode</span><b>{h(x.get("mode") or "-")}</b></div>
            <div><span>Prompt tok/s*</span><b>{fmt(x.get("prompt_tok_s_approx"),2)}</b></div>
            <div><span>Decode avg/request</span><b>{fmt(x.get("decode_avg_tok_s"),2)} tok/s</b></div>
            <div><span>GPU load avg/peak</span><b>{fmt(x.get("gpu_load_avg_pct"),1)} / {fmt(x.get("gpu_load_peak_pct"),1)}%</b></div>
            <div><span>VRAM peak</span><b>{fmt((x.get("gpu_vram_peak_mib") or 0)/1024,2)} GiB</b></div>
            <div><span>Fan peak</span><b>{fmt(x.get("gpu_fan_peak_pct"),0)}%</b></div>
            <div class="host-card"><span>Host platform</span><b>{h(_host_label(host))}</b><small>{h(host_source)}</small></div>
            <div class="host-card"><span>CPU</span><b>{h(host.get("cpu_model") or "-")}</b><small>{h(host.get("cpu_physical_cores") or "-")} cores / {h(host.get("cpu_logical_threads") or "-")} threads • {h(host.get("cpu_sockets") or "-")} socket(s)</small></div>
            <div class="host-card"><span>RAM</span><b>{fmt(host.get("ram_total_gib"),2)} GiB</b><small>{h(host.get("ram_type") or "type n/a")} • {h(host.get("ram_speed") or "speed n/a")}</small></div>
            <div class="host-card"><span>OS / kernel</span><b>{h(host.get("os") or "-")}</b><small>{h(host.get("kernel") or "-")}</small></div>
            <div><span>CPU avg</span><b>{fmt(x.get("cpu_avg_pct"),1)}%</b><small>activity, not token contribution</small></div>
            <div><span>RAM avg</span><b>{fmt(x.get("ram_avg_pct"),1)}%</b></div>
            <div><span>CPU role</span><b>Scheduling / tokenization / runtime</b><small>Exact % contribution to tok/s is not derivable from CPU utilization alone.</small></div>
            <div><span>Efficiency</span><b>{fmt(eff,3) if eff is not None else "-"} tok/s/W</b></div>
            <div><span>Comment</span><b>{h(x.get("comment") or "-")}</b></div>
          </div>
        </details>""")

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>AI6 vLLM selected benchmark report</title>
<style>
:root{{--bg:#0f0f10;--panel:#18191b;--line:#303236;--text:#ececec;--muted:#9ba0a6;--green:#73e28b;--blue:#79bfff;--detail-min:190px}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,Segoe UI,Arial,sans-serif}}
.wrap{{max-width:1800px;margin:0 auto;padding:24px}} h1{{margin:0 0 4px;font-size:24px}} .meta{{color:var(--muted);margin-bottom:18px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin:14px 0 18px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}} .card span,.card small{{display:block;color:var(--muted)}} .card b{{display:block;font-size:22px;margin:2px 0}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:10px;background:var(--panel)}} table{{border-collapse:collapse;width:100%;min-width:1450px}}
th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}} th{{position:sticky;top:0;background:#202226;color:#c9cdd1;font-size:12px;z-index:1}}
tbody tr:nth-child(even){{background:#141516}} tbody tr:hover{{background:#22252a}} .nowrap{{white-space:nowrap}} .sub{{color:var(--muted);font-size:11px;margin-top:2px}} .hot{{color:var(--green);font-weight:700}} .comment{{min-width:160px;max-width:260px;white-space:normal}} .profile{{display:inline-flex;align-items:center;justify-content:center;width:21px;height:21px;border-radius:999px;font-weight:800;margin-right:6px}} .profile.short{{color:#8ee7a0;background:#16311d;border:1px solid #2d7140}} .profile.medium{{color:#ffd56a;background:#332a11;border:1px solid #7f681f}} .profile.long{{color:#ff8c8c;background:#351818;border:1px solid #7d3131}} .profile-name{{color:var(--muted);font-size:11px}}
details{{margin-top:10px;background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:10px}} summary{{cursor:pointer;font-weight:700}}
.details-head{{display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap;margin-top:18px}} .details-head h2{{margin:0}} .detail-width-controls{{display:flex;gap:5px;align-items:center;color:var(--muted);font-size:11px}} .detail-width-controls button{{background:#17191c;color:#cfd3d7;border:1px solid var(--line);border-radius:6px;padding:5px 8px;cursor:pointer}} .detail-width-controls button.active{{color:var(--green);border-color:#3d7d4c;background:#172019}}
.detail-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(var(--detail-min),1fr));gap:8px;margin-top:10px}} .detail-grid>div{{background:#121315;border:1px solid #292b2f;border-radius:7px;padding:8px;min-width:0}} .detail-grid span{{display:block;color:var(--muted);font-size:11px}} .detail-grid b{{display:block;margin-top:2px;overflow-wrap:anywhere}} .detail-grid small{{display:block;color:#7f858b;font-size:10px;margin-top:3px;line-height:1.25}} .host-card{{border-color:#334250!important}}
.note{{color:var(--muted);font-size:12px;margin-top:12px}}
</style>
</head>
<body><div class="wrap">
<h1>AI6 vLLM selected benchmark report</h1>
<div class="meta">Generated {h(time.strftime('%Y-%m-%d %H:%M:%S %z'))}</div>
{highlights}
<div class="table-wrap"><table>
<thead><tr>
<th>#</th><th>Date</th><th>Model / quant</th><th>GPUs</th><th>PCIe</th><th>Profile</th><th>Temp</th><th>Thinking</th><th>Conc</th><th>PL total</th><th>Prompt</th><th>Out/req</th><th>TTFT avg/max</th><th>Wall</th><th>Aggregate</th><th>Per request</th><th>Power avg/peak</th><th>Temp</th><th>tok/s/W</th><th>Comment</th>
</tr></thead>
<tbody>{''.join(body_rows)}</tbody>
</table></div>
<div class="note">* Prompt tok/s is approximate because TTFT includes queueing and first-token overhead. CPU utilization is activity telemetry, not a direct percentage contribution to token generation. Legacy runs show the current host snapshot explicitly marked as not recorded at run time.</div>
<div class="details-head"><h2>Run details</h2><div class="detail-width-controls"><span>Card width</span><button type="button" onclick="setDetailWidth('compact')">Compact</button><button type="button" class="active" onclick="setDetailWidth('normal')">Normal</button><button type="button" onclick="setDetailWidth('wide')">Wide</button></div></div>
{''.join(detail_sections)}
<script>
function setDetailWidth(mode){
  const widths={compact:'150px',normal:'190px',wide:'270px'};
  document.documentElement.style.setProperty('--detail-min',widths[mode]||widths.normal);
  document.querySelectorAll('.detail-width-controls button').forEach(b=>b.classList.toggle('active',b.textContent.toLowerCase()===mode));
}
</script>
</div></body></html>"""


@base.app.post("/api/vllm/benchmark/meta")
def api_vllm_benchmark_meta(cmd: BenchmarkMetaCommand):
    row = _find_bench(cmd.run_id)
    if row is None:
        raise base.HTTPException(status_code=404, detail="benchmark run not found")
    if cmd.selected is not None:
        row["selected"] = bool(cmd.selected)
    if cmd.comment is not None:
        row["comment"] = cmd.comment.strip()
    _save_bench_results()
    return {"ok": True, "run": row}


@base.app.post("/api/vllm/benchmark/delete")
def api_vllm_benchmark_delete(cmd: BenchmarkDeleteCommand):
    ids = set(cmd.run_ids)
    before = len(_BENCH_RESULTS)
    _BENCH_RESULTS[:] = [x for x in _BENCH_RESULTS if x.get("run_id") not in ids]
    deleted = before - len(_BENCH_RESULTS)
    _save_bench_results()
    return {"ok": True, "deleted": deleted, "remaining": len(_BENCH_RESULTS)}


@base.app.get("/api/vllm/report/selected")
def api_vllm_selected_report(download: int = 0):
    rows = [x for x in _BENCH_RESULTS if x.get("selected")]
    if not rows:
        raise base.HTTPException(status_code=400, detail="no benchmark rows selected")
    if download:
        body = _combined_benchmark_report(rows)
        filename = "vllm-selected-" + time.strftime("%Y%m%d-%H%M%S") + ".md"
        return PlainTextResponse(
            body,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    return HTMLResponse(_combined_benchmark_report_html(rows))


@base.app.post("/api/vllm/git/upload-selected")
def api_vllm_git_upload_selected():
    selected = [x for x in _BENCH_RESULTS if x.get("selected")]
    if not selected:
        raise base.HTTPException(status_code=400, detail="no benchmark rows selected")

    rel_paths = []
    stamp = time.strftime("%Y-%m-%d")
    out_dir = _REPO_ROOT / "benchmarks" / "vllm-selected" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in selected:
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", row.get("model") or "model")
        safe_run = re.sub(r"[^A-Za-z0-9._-]+", "_", row.get("run_id") or str(int(time.time())))
        name = f"{safe_model}-{row.get('concurrency','x')}x{row.get('max_tokens','x')}-{safe_run}.md"
        path = out_dir / name
        path.write_text(_single_benchmark_report(row), encoding="utf-8")
        rel_paths.append(str(path.relative_to(_REPO_ROOT)))

    def git(*args, check=True):
        p = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), *args],
            capture_output=True, text=True, timeout=60,
        )
        if check and p.returncode != 0:
            raise RuntimeError((p.stderr or p.stdout or "git command failed").strip())
        return p

    try:
        git("add", "--", *rel_paths)
        diff = git("diff", "--cached", "--quiet", check=False)
        if diff.returncode == 0:
            return {"ok": True, "message": "Selected reports are already committed; nothing to upload.", "files": rel_paths}
        commit_msg = f"Add {len(rel_paths)} selected vLLM benchmark report{'s' if len(rel_paths) != 1 else ''}"
        git("commit", "-m", commit_msg)
        push = git("push")
        for row in selected:
            row["git_uploaded"] = True
            row["git_uploaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _save_bench_results()
        return {"ok": True, "message": (push.stdout or push.stderr or "Pushed to Git").strip(), "files": rel_paths}
    except Exception as e:
        raise base.HTTPException(status_code=500, detail=f"git upload failed: {e}")


@base.app.get("/api/vllm/report/run/{run_id}")
def api_vllm_run_report(run_id: str, download: int = 0):
    row = next((x for x in _BENCH_RESULTS if x.get("run_id") == run_id), None)
    if row is None:
        raise base.HTTPException(status_code=404, detail="benchmark run not found")
    body = _single_benchmark_report(row)
    headers = {}
    if download:
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", row.get("model") or "model")
        filename = f"vllm-{safe_model}-{row.get('concurrency','x')}x{row.get('max_tokens','x')}-{run_id}.md"
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return PlainTextResponse(body, media_type="text/markdown; charset=utf-8", headers=headers)


@base.app.get("/api/vllm/report")
def api_vllm_report():
    status = _vllm_status()
    stats = base.collect_stats()
    gpu_devices = ((stats.get("gpu") or {}).get("devices") or [])
    gpu = (gpu_devices or [None])[0] or {}
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
        f"- Quantization: {status.get('quantization') or '-'}",
        f"- vLLM version: {status.get('vllm_version') or '-'}",
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
        f"- GPU count: {len(gpu_devices)}",
        f"- vLLM selected GPUs: {', '.join('GPU'+str(i) for i in (status.get('gpu_ids') or [0]))}",
    ]
    for i, g in enumerate(gpu_devices):
        lines.append(
            f"- GPU{i} {g.get('name','-')}: load {g.get('utilization_pct','-')} %, "
            f"power {g.get('power_w','-')} W / PL {g.get('power_limit_w','-')} W, "
            f"VRAM {g.get('memory_used_mib','-')}/{g.get('memory_total_mib','-')} MiB, "
            f"temp {g.get('temperature_c','-')} C, fan {g.get('fan_pct','-')} %"
        )
    lines += [
        "",
        "## UI benchmark results",
        "",
        "| Concurrent | Prompt tok | Output/request | Total tokens | TTFT avg/max | Prompt tok/s* | Wall s | Aggregate tok/s | Per-request tok/s | Avg GPU load | Avg/peak power | Peak VRAM | Peak temp |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if _BENCH_RESULTS:
        for x in _BENCH_RESULTS:
            lines.append(
                f"| {x['concurrency']} | {x.get('prompt_tokens', '-')} | {x['max_tokens']} | {x['total_tokens']} | "
                f"{x.get('ttft_avg_s', 0):.3f}/{x.get('ttft_max_s', 0):.3f} s | "
                f"{x.get('prompt_tok_s_approx', 0):.2f} | "
                f"{x['wall_s']:.3f} | {x['aggregate_tok_s']:.2f} | "
                f"{x['per_request_min_tok_s']:.2f}-{x['per_request_max_tok_s']:.2f} | "
                f"{x.get('gpu_load_avg_pct', '-') if x.get('gpu_load_avg_pct') is not None else '-'}% | "
                f"{x.get('gpu_power_avg_w', '-') if x.get('gpu_power_avg_w') is not None else '-'} / "
                f"{x.get('gpu_power_peak_w', '-') if x.get('gpu_power_peak_w') is not None else '-'} W | "
                f"{round((x.get('gpu_vram_peak_mib') or 0)/1024, 2) if x.get('gpu_vram_peak_mib') is not None else '-'} GiB | "
                f"{x.get('gpu_temp_peak_c', '-') if x.get('gpu_temp_peak_c') is not None else '-'} C |"
            )
    else:
        lines.append("| - | - | - | - | - | - | - | - | no UI benchmark runs yet | - | - | - | - |")
    if _BENCH_RESULTS:
        lines += [
            "",
            "*Prompt tok/s is approximate: total prompt tokens divided by the slowest TTFT in the batch; TTFT includes queueing and first-token overhead.*",
            "",
            "## Benchmark telemetry details",
            "",
        ]
        for x in _BENCH_RESULTS:
            eff = None
            if x.get("gpu_power_avg_w") and x.get("aggregate_tok_s"):
                eff = x["aggregate_tok_s"] / x["gpu_power_avg_w"]
            lines += [
                f"### {x['concurrency']} concurrent × {x['max_tokens']} output tokens",
                f"- Model: {x.get('model') or '-'}",
                f"- Quantization: {x.get('quantization') or '-'}",
                f"- Context: {x.get('context') or '-'}",
                f"- Mode: {x.get('mode') or '-'}",
                f"- vLLM: {x.get('vllm_version') or '-'}",
                f"- GPUs: {_gpu_config_label(x)}",
                f"- Driver: {x.get('driver_version') or '-'}",
                f"- CUDA runtime: {(x.get('cuda') or {}).get('cuda_runtime') or '-'}",
                f"- Torch: {(x.get('cuda') or {}).get('torch') or '-'}",
                f"- Prompt profile ID: {x.get('prompt_profile_id') or '-'}",
                f"- Prompt set SHA256: {x.get('prompt_set_hash') or '-'}",
                f"- Temperature: {x.get('temperature', '-')}",
                f"- Thinking: {'ON' if x.get('enable_thinking') else 'OFF'}",
                f"- PCIe: {'; '.join('GPU'+str(p.get('index','?'))+': Gen'+str(p.get('gen_current') or '?')+' x'+str(p.get('width_current') or '?')+' (max Gen'+str(p.get('gen_max') or '?')+' x'+str(p.get('width_max') or '?')+')' for p in (x.get('pcie') or [])) or '-'}",
                f"- Prompt tokens total: {x.get('prompt_tokens', '-')}",
                f"- Output tokens total: {x.get('output_tokens', '-')}",
                f"- TTFT average: {x.get('ttft_avg_s', 0):.3f} s",
                f"- TTFT max: {x.get('ttft_max_s', 0):.3f} s",
                f"- Approx. prompt throughput: {x.get('prompt_tok_s_approx', 0):.2f} tok/s",
                f"- Average decode speed/request: {x.get('decode_avg_tok_s', 0):.2f} tok/s",
                f"- Aggregate output throughput: {x['aggregate_tok_s']:.2f} tok/s",
                f"- Average GPU load: {x.get('gpu_load_avg_pct') if x.get('gpu_load_avg_pct') is not None else '-'} %",
                f"- Peak GPU load: {x.get('gpu_load_peak_pct') if x.get('gpu_load_peak_pct') is not None else '-'} %",
                f"- Average GPU power: {x.get('gpu_power_avg_w') if x.get('gpu_power_avg_w') is not None else '-'} W",
                f"- Peak GPU power: {x.get('gpu_power_peak_w') if x.get('gpu_power_peak_w') is not None else '-'} W",
                f"- Power limit: {x.get('gpu_power_limit_w') if x.get('gpu_power_limit_w') is not None else '-'} W",
                f"- Peak VRAM: {round((x.get('gpu_vram_peak_mib') or 0)/1024, 2) if x.get('gpu_vram_peak_mib') is not None else '-'} GiB",
                f"- Peak temperature: {x.get('gpu_temp_peak_c') if x.get('gpu_temp_peak_c') is not None else '-'} C",
                f"- Peak fan: {x.get('gpu_fan_peak_pct') if x.get('gpu_fan_peak_pct') is not None else '-'} %",
                f"- Average CPU: {x.get('cpu_avg_pct') if x.get('cpu_avg_pct') is not None else '-'} %",
                f"- Average RAM: {x.get('ram_avg_pct') if x.get('ram_avg_pct') is not None else '-'} %",
                f"- Efficiency: {eff:.3f} aggregate tok/s/W" if eff is not None else "- Efficiency: -",
                f"- Telemetry samples: {x.get('sample_count', 0)}",
                "",
            ]
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


def _bench_sample():
    try:
        stats = base.collect_stats()
        devices = ((stats.get("gpu") or {}).get("devices") or [])
        return {
            "gpus": [
                {
                    "index": i,
                    "name": g.get("name"),
                    "gpu_load_pct": g.get("utilization_pct"),
                    "gpu_power_w": g.get("power_w"),
                    "gpu_power_limit_w": g.get("power_limit_w"),
                    "gpu_vram_used_mib": g.get("memory_used_mib"),
                    "gpu_vram_total_mib": g.get("memory_total_mib"),
                    "gpu_temp_c": g.get("temperature_c"),
                    "gpu_fan_pct": g.get("fan_pct"),
                }
                for i, g in enumerate(devices)
            ],
            "cpu_pct": (stats.get("cpu") or {}).get("usage_pct"),
            "ram_pct": (stats.get("memory") or {}).get("usage_pct"),
        }
    except Exception:
        return {}


def _summarize_bench_samples(samples, gpu_ids=None):
    gpu_ids = list(gpu_ids or [0])

    def vals(key):
        out = []
        for s in samples:
            v = s.get(key)
            if isinstance(v, (int, float)):
                out.append(float(v))
        return out

    def avg_list(items):
        return round(sum(items) / len(items), 2) if items else None

    def peak_list(items):
        return round(max(items), 2) if items else None

    gpu_rows = []
    for gpu_id in gpu_ids:
        per = []
        for s in samples:
            g = next((x for x in (s.get("gpus") or []) if x.get("index") == gpu_id), None)
            if g:
                per.append(g)
        def gv(key):
            return [float(x[key]) for x in per if isinstance(x.get(key), (int, float))]
        first = per[0] if per else {}
        gpu_rows.append({
            "index": gpu_id,
            "name": first.get("name"),
            "load_avg_pct": avg_list(gv("gpu_load_pct")),
            "load_peak_pct": peak_list(gv("gpu_load_pct")),
            "power_avg_w": avg_list(gv("gpu_power_w")),
            "power_peak_w": peak_list(gv("gpu_power_w")),
            "power_limit_w": peak_list(gv("gpu_power_limit_w")),
            "vram_peak_mib": peak_list(gv("gpu_vram_used_mib")),
            "vram_total_mib": peak_list(gv("gpu_vram_total_mib")),
            "temp_peak_c": peak_list(gv("gpu_temp_c")),
            "fan_peak_pct": peak_list(gv("gpu_fan_pct")),
        })

    total_power_samples = []
    total_vram_samples = []
    mean_load_samples = []
    max_temp_samples = []
    for s in samples:
        gs = [g for g in (s.get("gpus") or []) if g.get("index") in gpu_ids]
        powers = [float(g["gpu_power_w"]) for g in gs if isinstance(g.get("gpu_power_w"), (int,float))]
        vrams = [float(g["gpu_vram_used_mib"]) for g in gs if isinstance(g.get("gpu_vram_used_mib"), (int,float))]
        loads = [float(g["gpu_load_pct"]) for g in gs if isinstance(g.get("gpu_load_pct"), (int,float))]
        temps = [float(g["gpu_temp_c"]) for g in gs if isinstance(g.get("gpu_temp_c"), (int,float))]
        if powers: total_power_samples.append(sum(powers))
        if vrams: total_vram_samples.append(sum(vrams))
        if loads: mean_load_samples.append(sum(loads) / len(loads))
        if temps: max_temp_samples.append(max(temps))

    return {
        "gpu_count": len(gpu_ids),
        "gpu_ids": gpu_ids,
        "gpu_devices": gpu_rows,
        "gpu_load_avg_pct": avg_list(mean_load_samples),
        "gpu_load_peak_pct": peak_list(mean_load_samples),
        "gpu_power_avg_w": avg_list(total_power_samples),
        "gpu_power_peak_w": peak_list(total_power_samples),
        "gpu_power_limit_w": round(sum(x.get("power_limit_w") or 0 for x in gpu_rows), 2) if gpu_rows else None,
        "gpu_vram_peak_mib": peak_list(total_vram_samples),
        "gpu_temp_peak_c": peak_list(max_temp_samples),
        "gpu_fan_peak_pct": peak_list([x.get("fan_peak_pct") for x in gpu_rows if isinstance(x.get("fan_peak_pct"), (int,float))]),
        "cpu_avg_pct": avg_list(vals("cpu_pct")),
        "ram_avg_pct": avg_list(vals("ram_pct")),
        "sample_count": len(samples),
    }


def _benchmark_prompt(i, profile):
    base_prompt = (
        f"Task {i}. Write production-quality Python code with type hints, "
        "async concurrency, retries, timeouts, logging, metrics, error handling, "
        "tests, and graceful shutdown. Explain important design choices."
    )
    profile = (profile or "short").lower()
    if profile == "short":
        return base_prompt
    filler = (
        " Requirements: validate inputs; avoid global mutable state; use clear abstractions; "
        "document failure modes; consider cancellation, backpressure, resource cleanup, "
        "performance, observability, deterministic behavior, and maintainability."
    )
    target_chars = 1900 if profile == "medium" else 7600 if profile == "long" else 1900
    text = base_prompt
    while len(text) < target_chars:
        text += filler
    return text[:target_chars]


@base.app.post("/api/vllm/benchmark")
def api_vllm_benchmark(cmd: VllmBenchmarkCommand):
    status = _vllm_status()
    if not status["ready"]:
        raise base.HTTPException(status_code=409, detail="vLLM server is not ready")
    port = status["port"]
    model = status["model"]
    if cmd.prompt_profile not in ("short", "medium", "long"):
        raise base.HTTPException(status_code=400, detail="prompt_profile must be short, medium, or long")
    prompts = [
        _benchmark_prompt(i, cmd.prompt_profile)
        for i in range(1, cmd.concurrency + 1)
    ]
    prompt_profile_id = f"{cmd.prompt_profile}-v1"
    prompt_set_hash = hashlib.sha256(
        json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    prompt_first_hash = hashlib.sha256(prompts[0].encode("utf-8")).hexdigest() if prompts else None
    started = time.perf_counter()
    samples = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=cmd.concurrency) as pool:
        futures = [
            pool.submit(_bench_one, port, model, prompt, cmd.max_tokens, cmd.temperature, cmd.enable_thinking)
            for prompt in prompts
        ]
        while True:
            samples.append(_bench_sample())
            if all(f.done() for f in futures):
                break
            time.sleep(0.5)
        results = [f.result() for f in futures]
    wall = time.perf_counter() - started
    total_output_tokens = sum(x["completion_tokens"] for x in results)
    total_prompt_tokens = sum(x["prompt_tokens"] for x in results)
    total_tokens = total_prompt_tokens + total_output_tokens
    per_request = [x["e2e_tok_s"] for x in results]
    decode_rates = [x["decode_tok_s"] for x in results]
    ttfts = [x["ttft_s"] for x in results]
    # Approximate prompt-prefill throughput for the batch. TTFT contains queueing
    # and first-token overhead, so keep it explicitly labelled as approximate.
    prompt_window = max(ttfts) if ttfts else 0.0
    prompt_tok_s_approx = total_prompt_tokens / prompt_window if prompt_window > 0 else 0.0
    row = {
        "run_id": f"{int(time.time())}-{cmd.concurrency}-{cmd.max_tokens}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "concurrency": cmd.concurrency,
        "max_tokens": cmd.max_tokens,
        "temperature": cmd.temperature,
        "enable_thinking": bool(cmd.enable_thinking),
        "prompt_profile": cmd.prompt_profile,
        "prompt_profile_id": prompt_profile_id,
        "prompt_set_hash": prompt_set_hash,
        "prompt_first_hash": prompt_first_hash,
        "prompt_tokens": total_prompt_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "wall_s": round(wall, 3),
        "aggregate_tok_s": round(total_output_tokens / wall, 2) if wall > 0 else 0,
        "prompt_tok_s_approx": round(prompt_tok_s_approx, 2),
        "ttft_avg_s": round(sum(ttfts) / len(ttfts), 3) if ttfts else 0,
        "ttft_max_s": round(max(ttfts), 3) if ttfts else 0,
        "decode_avg_tok_s": round(sum(decode_rates) / len(decode_rates), 2) if decode_rates else 0,
        "per_request_min_tok_s": round(min(per_request), 2) if per_request else 0,
        "per_request_max_tok_s": round(max(per_request), 2) if per_request else 0,
        "model": Path(status.get("model_path") or status.get("model") or "").name or None,
        "quantization": status.get("quantization"),
        "context": status.get("max_model_len"),
        "mode": "eager" if status.get("enforce_eager") else "compiled/graphs",
        "vllm_version": status.get("vllm_version"),
        "host": _host_platform_info(),
        **_system_cuda_info(),
        "pcie": [_pcie_info_for_gpu(i) for i in (status.get("gpu_ids") or [0])],
        **_summarize_bench_samples(samples, status.get("gpu_ids") or [0]),
    }
    row.setdefault("selected", False)
    row.setdefault("comment", "")
    _BENCH_RESULTS.append(row)
    del _BENCH_RESULTS[:-500]
    _save_bench_results()
    return {"ok": True, "result": row}


def install():
    dashboard = base.DASHBOARD

    nav = """<div class="top-tabs">
<button id="tabMonitorBtn" class="tab-btn active" onclick="showTopTab('monitor')">Monitor</button>
<button id="tabVllmBtn" class="tab-btn" onclick="showTopTab('vllm')">vLLM</button>
<button id="tabBenchsBtn" class="tab-btn" onclick="showTopTab('benchs')">Benchs</button>
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
      <label>Model<div class="vllm-model-row"><select id="vllmModel"></select><button type="button" title="Reload model list" onclick="loadVllmModels(true)">↻</button></div></label>
      <label>Port<input id="vllmPort" type="number" value="8012" min="1024" max="65535"></label>
      <label>Context<input id="vllmCtx" type="number" value="4096" min="256"></label>
      <label>GPU memory<input id="vllmMem" type="number" value="0.90" min="0.10" max="0.99" step="0.01"></label>
      <label class="vllm-check"><input id="vllmEager" type="checkbox"> Enforce eager</label>
      <label>GPUs<div id="vllmGpuSelect" class="vllm-gpu-select"><span class="muted">detecting…</span></div></label>
      <label>Power limit
        <div class="vllm-pl-row">
          <select id="vllmPlPreset" onchange="applyPlPresetToInput()">
            <option value="130">130 W</option>
            <option value="140">140 W</option>
            <option value="150" selected>150 W</option>
            <option value="165">165 W</option>
            <option value="184">184 W</option>
            <option value="200">200 W</option>
            <option value="220">220 W</option>
            <option value="custom">Custom</option>
          </select>
          <input id="vllmPlCustom" type="number" value="150" min="1" max="1000" step="1">
          <button type="button" onclick="setGpuPowerLimit()">Set</button>
        </div>
        <small id="vllmPlInfo">GPU #0 power limit</small>
      </label>
    </div>
    <div id="vllmLoadWrap" class="vllm-load-wrap" style="display:none">
      <div class="vllm-load-line"><span id="vllmLoadText">Loading model…</span><span id="vllmLoadPct">0%</span></div>
      <div class="vllm-load-bar"><div id="vllmLoadFill" class="vllm-load-fill"></div></div>
      <div id="vllmLoadSub" class="vllm-load-sub">Waiting for model initialization…</div>
    </div>
    <div class="vllm-actions">
      <button onclick="startVllm()">Start</button>
      <button onclick="controlVllm('stop')">Stop</button>
      <button onclick="applyRestartVllm()">Apply / Restart</button>
      <button onclick="downloadVllmReport()">Download report</button>
      <button onclick="toggleVllmLogs()">Logs</button>
      <span id="vllmMsg" class="muted"></span>
    </div>
    <div class="vllm-cards">
      <div class="worker-stat">PID<b id="vllmPid">—</b></div>
      <div class="worker-stat">Running model<b id="vllmLiveModel">—</b></div>
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
    </div>
    <div id="vllmGpuGrid" class="vllm-gpu-grid"></div>
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
      <div><div class="section-title">vLLM Batch Benchmark</div><div class="section-sub">Configurable continuous-batching benchmark</div></div>
    </div>
    <div class="vllm-bench-meta">
      <span>Model <b id="benchModel">—</b></span>
      <span>Quant <b id="benchQuant">—</b></span>
      <span>Context <b id="benchCtx">—</b></span>
      <span>Mode <b id="benchMode">—</b></span>
      <span>PL <b id="benchPl">—</b></span>
      <span>GPUs <b id="benchGpu">—</b></span>
      <span>vLLM <b id="benchVllm">—</b></span>
      <span>Thinking <b id="benchThinkingState">OFF</b></span>
    </div>
    <div class="vllm-bench-config">
      <label>Output/request
        <select id="benchOutTokens">
          <option value="128">128</option>
          <option value="256">256</option>
          <option value="512" selected>512</option>
          <option value="1024">1024</option>
          <option value="2048">2048</option>
          <option value="4096">4096</option>
        </select>
      </label>
      <label>Prompt size
        <select id="benchPromptProfile">
          <option value="short" selected>Short (~40 tok)</option>
          <option value="medium">Medium (~500 tok)</option>
          <option value="long">Long (~2000 tok)</option>
        </select>
      </label>
      <label>Temperature
        <input id="benchTemperature" type="number" value="0" min="0" max="2" step="0.1">
      </label>
      <label>Thinking
        <select id="benchThinking" onchange="syncBenchThinkingBadge()">
          <option value="off" selected>OFF</option>
          <option value="on">ON</option>
        </select>
      </label>
      <label>Custom conc
        <div class="bench-custom-conc"><input id="benchCustomConc" type="number" value="12" min="1" max="128"><button onclick="runVllmBench(Number(benchCustomConc.value))">Run</button></div>
      </label>
    </div>
    <div class="vllm-bench-toolbar">
      <button class="git-upload-btn" onclick="uploadSelectedBenchmarks()">↑ Upload selected to Git</button>
      <button onclick="setAllBenchSelection(true)">Select all</button>
      <button onclick="setAllBenchSelection(false)">Clear</button>
      <button class="delete-selected-btn" onclick="deleteSelectedBenchmarks('main')">Del selected</button>
      <span id="vllmGitMsg" class="muted"></span>
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
        <thead><tr><th>✓</th><th>Concurrent</th><th>Profile</th><th>Prompt hash</th><th>Prompt tok</th><th>Out/req</th><th>Total tok</th><th>TTFT avg/max</th><th>Prompt tok/s*</th><th>Wall</th><th>Aggregate</th><th>Per request</th><th>Model / quant</th><th>Comment</th></tr></thead>
        <tbody id="vllmBenchRows"><tr><td colspan="5" class="muted">No UI benchmark runs yet</td></tr></tbody>
      </table>
    </div>
  </div>
</div>

<div id="benchsTab" style="display:none">
  <div class="section">
    <div class="section-head">
      <div>
        <div class="section-title">Benchmark History</div>
        <div class="section-sub">Persistent results from <span id="benchStorePath">~/.local/state/ai6-monitor/vllm-benchmarks.json</span></div>
      </div>
      <div id="benchHistoryCount" class="muted">0 runs</div>
    </div>

    <div id="benchBestCards" class="bench-best-cards">
      <div class="bench-best-card"><span>Best aggregate</span><b id="benchBestAgg">—</b><small id="benchBestAggSub">visible results</small></div>
      <div class="bench-best-card"><span>Best efficiency</span><b id="benchBestEff">—</b><small id="benchBestEffSub">visible results</small></div>
      <div class="bench-best-card"><span>Best balanced</span><b id="benchBestScore">—</b><small id="benchBestScoreSub">visible results</small></div>
    </div>
    <div class="bench-history-toolbar">
      <button class="git-upload-btn" onclick="uploadSelectedBenchmarks()">↑ Upload selected to Git</button>
      <button onclick="setAllHistorySelection(true)">Select visible</button>
      <button onclick="setAllHistorySelection(false)">Clear visible</button>
      <button onclick="clearBenchFilters()">Clear filters</button>
      <button class="delete-selected-btn" onclick="deleteSelectedBenchmarks('history')">Del selected</button>
      <span class="combined-report-actions">
        <a class="run-dl" title="Download combined report for selected rows" href="/api/vllm/report/selected?download=1">↓</a>
        <a class="run-open" title="Open combined report for selected rows" target="_blank" href="/api/vllm/report/selected">↗</a>
      </span>
      <span id="benchHistoryMsg" class="muted"></span>
    </div>
    <div class="section-sub" style="margin:-4px 0 9px">Rating = balanced score inside comparable workload peers (aggregate 35% • per-request 25% • efficiency 20% • TTFT 20%). Recommendations are workload hints, not model-quality scores.</div>

    <div class="bench-filter-grid">
      <label>Date<select id="bfDate" onchange="renderBenchHistory()"><option value="">all</option></select></label>
      <label>Model<select id="bfModel" onchange="renderBenchHistory()"><option value="">all</option></select></label>
      <label>Quant<select id="bfQuant" onchange="renderBenchHistory()"><option value="">all</option></select></label>
      <label>Conc<select id="bfConc" onchange="renderBenchHistory()"><option value="">all</option></select></label>
      <label>Out/req<select id="bfOut" onchange="renderBenchHistory()"><option value="">all</option></select></label>
      <label>Prompt size<select id="bfPromptProfile" onchange="renderBenchHistory()"><option value="">all</option></select></label>
      <label>Temp<select id="bfTemp" onchange="renderBenchHistory()"><option value="">all</option></select></label>
      <label>Thinking<select id="bfThinking" onchange="renderBenchHistory()"><option value="">all</option><option value="off">OFF</option><option value="on">ON</option></select></label>
      <label>PCIe<select id="bfPcie" onchange="renderBenchHistory()"><option value="">all</option></select></label>
      <label>GPUs<select id="bfGpus" onchange="renderBenchHistory()"><option value="">all</option></select></label>
      <label>PL<select id="bfPl" onchange="renderBenchHistory()"><option value="">all</option></select></label>
      <label>Git
        <select id="bfGit" onchange="renderBenchHistory()">
          <option value="">all</option>
          <option value="yes">uploaded</option>
          <option value="no">not uploaded</option>
        </select>
      </label>
      <label>Comment<input id="bfComment" placeholder="text…" oninput="renderBenchHistory()"></label>
      <label>Min agg<input id="bfAgg" placeholder="700" type="number" oninput="renderBenchHistory()"></label>
    </div>

    <div class="vllm-table-wrap bench-history-wrap">
      <table class="vllm-table bench-history-table">
        <thead>
          <tr>
            <th>✓</th><th>Date</th><th>Model / quant</th><th>GPUs</th><th>PCIe</th><th>Think</th><th>Profile</th><th>Prompt hash</th><th>Conc</th>
            <th>Prompt</th><th>Out/req</th><th>TTFT avg/max</th><th>Wall</th>
            <th>Aggregate</th><th>Per req</th><th>PL</th><th>Power avg/peak</th>
            <th>Temp</th><th>Rating</th><th>Recommended</th><th>Git</th><th>Report</th><th>Comment</th>
          </tr>
        </thead>
        <tbody id="benchHistoryRows"><tr><td colspan="23" class="muted">Loading benchmark history…</td></tr></tbody>
      </table>
    </div>
  </div>
</div>
'''
    dashboard = dashboard.replace("<script>", vllm_html + "\n<script>", 1)

    css = r'''
.top-tabs{display:flex;gap:8px;margin:14px 0 2px}.tab-btn{padding:8px 16px}.tab-btn.active{border-color:#7be495;color:#7be495;background:#172019}
.vllm-form{display:grid;grid-template-columns:minmax(250px,2fr) repeat(3,minmax(105px,1fr)) minmax(120px,1fr) minmax(300px,1.5fr);gap:8px;align-items:end}
.vllm-form label{font-size:11px;color:#aaa}.vllm-gpu-select{display:flex;gap:5px;flex-wrap:wrap;margin-top:4px}.vllm-gpu-choice{display:flex!important;align-items:center;gap:3px;background:#151515;border:1px solid #333;border-radius:6px;padding:5px 7px!important;color:#ddd!important;font-size:10px!important}.vllm-gpu-choice input{width:auto!important;margin:0!important}.vllm-gpu-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:8px;margin-top:8px}.vllm-gpu-card{background:#171717;border:1px solid #303030;border-radius:9px;padding:9px}.vllm-gpu-card-head{display:flex;justify-content:space-between;gap:8px}.vllm-gpu-card-head b{font-size:12px}.vllm-gpu-card-head span{font-size:10px;color:#999}.vllm-gpu-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:7px}.vllm-gpu-stats div{font-size:9px;color:#999}.vllm-gpu-stats b{display:block;color:#eee;font-size:13px}.vllm-gpu-stats b.ok{color:var(--ok)}.vllm-gpu-stats b.warn{color:var(--warn)}.vllm-gpu-stats b.bad{color:var(--bad)}.vllm-gpu-card.selected{border-color:#3d7d4c;box-shadow:0 0 0 1px rgba(115,226,139,.12) inset}.vllm-form select,.vllm-form input{display:block;width:100%;margin-top:4px;padding:7px}.vllm-model-row{display:grid;grid-template-columns:minmax(0,1fr) 34px;gap:5px;align-items:end}.vllm-model-row button{height:32px;padding:0}.vllm-pl-row{display:grid;grid-template-columns:90px 90px 58px;gap:5px;align-items:end}.vllm-pl-row select,.vllm-pl-row input{margin-top:4px!important}.vllm-pl-row button{padding:7px 8px}.vllm-form label small{display:block;margin-top:3px;color:#777;font-size:9px}.vllm-check{display:flex!important;align-items:center;gap:7px;padding:7px 4px}.vllm-check input{width:auto!important;margin:0!important}
.vllm-load-wrap{margin-top:10px}.vllm-load-line{display:flex;justify-content:space-between;gap:10px;font-size:11px;color:#bbb}.vllm-load-bar{height:9px;background:#2b2b2b;border:1px solid #383838;border-radius:6px;overflow:hidden;margin-top:5px}.vllm-load-fill{height:100%;width:0%;background:#8a8a8a;transition:width .35s ease}.vllm-load-sub{font-size:10px;color:#888;margin-top:4px}
.vllm-actions,.vllm-bench-buttons{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin-top:10px}.vllm-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin-top:12px}.vllm-cards .worker-stat b{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.vllm-log-wrap{margin-top:10px;background:#101010;border:1px solid #303030;border-radius:8px;padding:8px}.vllm-log-head{display:flex;justify-content:space-between;align-items:center;gap:8px;font-size:10px;color:#888}.vllm-log-head button{padding:4px 8px;font-size:10px}.vllm-log-wrap pre{margin:7px 0 0;max-height:300px;overflow:auto;white-space:pre-wrap;word-break:break-word;font-size:10px;line-height:1.35;color:#cfcfcf}
.vllm-host-strip{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:7px;margin-top:9px}.vllm-mini{background:#171717;border:1px solid #303030;border-radius:8px;padding:7px 9px;font-size:10px;color:#aaa;min-width:0}.vllm-mini b{display:block;margin-top:2px;font-size:17px;line-height:1.15;color:#eee;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.vllm-mini b.ok{color:var(--ok)}.vllm-mini b.warn{color:var(--warn)}.vllm-mini b.bad{color:var(--bad)}.vllm-mini small{display:block;margin-top:2px;color:#888;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.vllm-prompt{width:100%;min-height:120px;resize:vertical;background:#111;color:#eee;border:1px solid #444;border-radius:8px;padding:10px;font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace}.vllm-prompt-actions{display:flex;gap:8px;align-items:end;flex-wrap:wrap;margin-top:8px}.vllm-prompt-actions label{font-size:10px;color:#aaa}.vllm-prompt-actions input{display:block;width:100px;margin-top:3px;padding:6px}.vllm-prompt-output{margin:10px 0 0;max-height:420px;overflow:auto;white-space:pre-wrap;word-break:break-word;background:#101010;border:1px solid #303030;border-radius:8px;padding:10px;font-size:11px;line-height:1.4;color:#ddd}
.prompt-badge{display:inline-flex;align-items:center;justify-content:center;min-width:20px;height:20px;padding:0 6px;border-radius:999px;font-size:10px;font-weight:800;margin-right:5px;border:1px solid transparent}.prompt-badge.s{color:#8ee7a0;background:#16311d;border-color:#2d7140}.prompt-badge.m{color:#ffd56a;background:#332a11;border-color:#7f681f}.prompt-badge.l{color:#ff8c8c;background:#351818;border-color:#7d3131}
.vllm-bench-meta{display:flex;gap:7px;flex-wrap:wrap;margin:2px 0 10px}.vllm-bench-meta span{background:#171717;border:1px solid #303030;border-radius:7px;padding:5px 7px;font-size:10px;color:#999}.vllm-bench-meta b{color:#eee;font-weight:700;margin-left:3px}
.vllm-bench-config{display:flex;gap:8px;align-items:end;flex-wrap:wrap;margin:0 0 10px}.vllm-bench-config label{font-size:10px;color:#999}.vllm-bench-config select,.vllm-bench-config input{display:block;margin-top:3px;padding:6px 8px;background:#111;color:#ddd;border:1px solid #3a3a3a;border-radius:6px}.bench-custom-conc{display:flex;gap:4px}.bench-custom-conc input{width:72px}.bench-custom-conc button{padding:6px 10px}
.vllm-bench-toolbar{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:0 0 8px}.git-upload-btn{border-color:#2f7540!important;color:#72e28a!important;background:#132519!important}.bench-select{width:16px;height:16px}.bench-comment{width:150px;max-width:22vw;background:#111;color:#ddd;border:1px solid #3a3a3a;border-radius:5px;padding:4px 6px;font-size:10px}
.run-report-actions{display:inline-flex;gap:4px;margin-left:6px;vertical-align:middle}.run-report-actions a{display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;border-radius:5px;text-decoration:none;font-weight:800;font-size:12px;border:1px solid #3a3a3a}.run-report-actions a.run-dl{color:#72e28a;border-color:#2f7540;background:#132519}.run-report-actions a.run-open{color:#76b9ff;border-color:#2d5f91;background:#122235}.run-report-actions a:hover{filter:brightness(1.2)}
.bench-best-cards{display:grid;grid-template-columns:repeat(3,minmax(180px,1fr));gap:8px;margin:10px 0 12px}.bench-best-card{background:#141719;border:1px solid #35523c;border-radius:10px;padding:10px 12px;box-shadow:0 0 0 1px rgba(115,226,139,.05) inset}.bench-best-card span{display:block;color:#9ca3a8;font-size:10px}.bench-best-card b{display:block;color:#7be495;font-size:20px;line-height:1.25;margin:2px 0}.bench-best-card small{color:#7d858b;font-size:9px}.bench-best-card.best-eff{border-color:#6b5a27}.bench-best-card.best-eff b{color:#ffd56a}.bench-best-card.best-score{border-color:#315d7b}.bench-best-card.best-score b{color:#79bfff}
.bench-history-toolbar{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:0 0 10px}.delete-selected-btn{border-color:#7a3434!important;color:#ff8585!important;background:#2a1515!important}.combined-report-actions{display:inline-flex;gap:5px;margin-left:2px}.combined-report-actions a{display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:6px;text-decoration:none;font-weight:900;font-size:15px;border:1px solid #3a3a3a}.combined-report-actions a.run-dl{color:#72e28a;border-color:#2f7540;background:#132519}.combined-report-actions a.run-open{color:#76b9ff;border-color:#2d5f91;background:#122235}.bench-filter-grid{display:grid;grid-template-columns:repeat(13,minmax(90px,1fr));gap:6px;margin-bottom:8px}.bench-filter-grid label{font-size:9px;color:#999}.bench-filter-grid input,.bench-filter-grid select{display:block;width:100%;margin-top:3px;padding:5px 6px;background:#111;color:#ddd;border:1px solid #3a3a3a;border-radius:5px;font-size:10px}.bench-history-wrap{max-height:68vh}.bench-history-table{min-width:1500px}.git-state-ok{color:#72e28a;font-weight:700}.git-state-no{color:#888}
.bench-history-table tr.row-best-agg td{background:rgba(61,125,76,.10)}.bench-history-table tr.row-best-eff td{box-shadow:inset 0 1px 0 rgba(255,213,106,.08),inset 0 -1px 0 rgba(255,213,106,.08)}.bench-history-table td.metric-good{color:#7be495;font-weight:700}.bench-history-table td.metric-warn{color:#ffd56a}.bench-history-table td.metric-bad{color:#ff8c8c}.rating-pill{display:inline-flex;align-items:center;gap:4px;border-radius:999px;padding:3px 7px;font-weight:800;font-size:10px;border:1px solid #3a3a3a}.rating-pill.r-high{color:#7be495;background:#142719;border-color:#326a40}.rating-pill.r-mid{color:#ffd56a;background:#2b2513;border-color:#6e5b21}.rating-pill.r-low{color:#ff9a9a;background:#2b1717;border-color:#6f3030}.rec-badges{display:flex;gap:3px;flex-wrap:wrap;min-width:120px}.rec-tag{display:inline-flex;border-radius:999px;padding:2px 6px;font-size:9px;border:1px solid #394047;color:#cdd2d6;background:#171a1d}.rec-tag.interactive{color:#7be495;border-color:#326a40;background:#142719}.rec-tag.batch{color:#79bfff;border-color:#315d7b;background:#14212a}.rec-tag.longctx{color:#d8a6ff;border-color:#65437c;background:#24172d}.rec-tag.coding{color:#ffd56a;border-color:#6e5b21;background:#2b2513}
.vllm-table-wrap{overflow:auto;margin-top:10px}.vllm-table th,.vllm-table td{text-align:left;padding:7px 8px;border-bottom:1px solid #2d2d2d}.vllm-table th{color:#bbb;font-size:11px}.vllm-table td{font-size:12px}
@media(max-width:900px){.vllm-form{grid-template-columns:1fr 1fr}.vllm-host-strip{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:560px){.vllm-host-strip{grid-template-columns:repeat(2,minmax(0,1fr))}}
'''
    dashboard = dashboard.replace("</style>", css + "\n</style>", 1)

    js = r'''
let vllmModelsLoaded=false;
let vllmPrevMetrics=null;
let vllmPrevTs=null;

function showTopTab(which){
 const mon=document.getElementById('monitorTab'),vl=document.getElementById('vllmTab'),bh=document.getElementById('benchsTab');
 const mb=document.getElementById('tabMonitorBtn'),vb=document.getElementById('tabVllmBtn'),bb=document.getElementById('tabBenchsBtn');
 mon.style.display=which==='monitor'?'block':'none';
 vl.style.display=which==='vllm'?'block':'none';
 bh.style.display=which==='benchs'?'block':'none';
 mb.classList.toggle('active',which==='monitor');
 vb.classList.toggle('active',which==='vllm');
 bb.classList.toggle('active',which==='benchs');
 if(which==='vllm'){loadVllmModels();refreshGpuPowerLimitInfo();refreshVllm();}
 if(which==='benchs'){loadBenchHistory();}
}

function selectedVllmGpuIds(){
 return [...document.querySelectorAll('.vllm-gpu-choice input:checked')].map(x=>Number(x.value)).filter(Number.isFinite);
}

function renderGpuSelector(devices,runningIds){
 const box=document.getElementById('vllmGpuSelect');
 if(!box)return;
 const current=selectedVllmGpuIds();
 const selected=current.length?current:(runningIds&&runningIds.length?runningIds:[0]);
 box.innerHTML=(devices||[]).map((g,i)=>{
   const checked=selected.includes(i)?' checked':'';
   return '<label class="vllm-gpu-choice"><input type="checkbox" value="'+i+'"'+checked+'>#'+i+' '+escHtml(g.name||'GPU')+'</label>';
 }).join('')||'<span class="muted">No GPUs</span>';
}

function renderVllmGpuGrid(devices,selectedIds){
 const grid=document.getElementById('vllmGpuGrid');
 if(!grid)return;
 grid.innerHTML=(devices||[]).map((g,i)=>{
   const selected=(selectedIds||[]).includes(i);
   const load=g.utilization_pct??0;
   const temp=g.temperature_c;
   const used=(g.memory_used_mib||0)/1024,total=(g.memory_total_mib||0)/1024;
   const power=Number(g.power_w||0),pl=Number(g.power_limit_w||0);
   const vramRatio=total?used/total:0;
   const powerRatio=pl?power/pl:0;

   const loadClass=load>=90?'ok':load>=50?'warn':'';
   const powerClass=powerRatio>=1.0?'bad':powerRatio>=0.90?'warn':'ok';
   const vramClass=vramRatio>=0.95?'bad':vramRatio>=0.85?'warn':'ok';
   const tempClass=temp==null?'':cls(temp);

   return '<div class="vllm-gpu-card'+(selected?' selected':'')+'">'+
     '<div class="vllm-gpu-card-head"><b>#'+i+' '+escHtml(g.name||'GPU')+'</b><span>'+(selected?'vLLM selected':'available')+'</span></div>'+
     '<div class="vllm-gpu-stats">'+
       '<div>Load<b class="'+loadClass+'">'+load+'%</b></div>'+
       '<div>Power<b class="'+powerClass+'">'+(g.power_w??'?')+' W</b><span> / '+(g.power_limit_w??'?')+' W</span></div>'+
       '<div>VRAM<b class="'+vramClass+'">'+used.toFixed(2)+' / '+total.toFixed(2)+' GiB</b></div>'+
       '<div>Temp<b class="'+tempClass+'">'+(temp??'?')+'°C</b><span> fan '+(g.fan_pct??'?')+'%</span></div>'+
     '</div></div>';
 }).join('');
}

async function loadVllmModels(force=false){
 if(vllmModelsLoaded&&!force&&document.getElementById('vllmModel')?.options?.length)return;
 try{
  const r=await fetch('/api/vllm/models',{cache:'no-store'});const d=await r.json();
  if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  const el=document.getElementById('vllmModel');
  const previous=el.value;
  const models=d.models||[];
  if(!models.length){
    el.innerHTML='<option value="">No models found</option>';
    vllmModelsLoaded=false;
    vllmMsg.textContent='No models found in ~/models/vllm';
    return;
  }
  el.innerHTML=models.map(m=>'<option value="'+m.path.replace(/"/g,'&quot;')+'">'+m.label+'</option>').join('');
  const previousIndex=models.findIndex(m=>m.path===previous);
  if(previousIndex>=0)el.selectedIndex=previousIndex;
  else{
    const preferred=models.findIndex(m=>m.label.toLowerCase().includes('qwen3-4b-awq'));
    if(preferred>=0)el.selectedIndex=preferred;
  }
  vllmModelsLoaded=true;
 }catch(e){
  vllmModelsLoaded=false;
  vllmMsg.textContent='Model list error: '+e.message;
 }
}

function applyPlPresetToInput(){
 const sel=document.getElementById('vllmPlPreset');
 const inp=document.getElementById('vllmPlCustom');
 if(sel.value!=='custom')inp.value=sel.value;
}

async function refreshGpuPowerLimitInfo(){
 try{
  const r=await fetch('/api/gpu/power-limit?gpu=0');const d=await r.json();
  if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  vllmPlInfo.textContent='Current '+d.current_w.toFixed(0)+' W • default '+d.default_w.toFixed(0)+' W • range '+d.min_w.toFixed(0)+'–'+d.max_w.toFixed(0)+' W';
  vllmPlCustom.min=d.min_w;vllmPlCustom.max=d.max_w;
 }catch(e){vllmPlInfo.textContent='Power limit info unavailable';}
}

async function setGpuPowerLimit(){
 const watts=Number(vllmPlCustom.value);
 if(!Number.isFinite(watts)){vllmMsg.textContent='Invalid power limit';return;}
 vllmMsg.textContent='Setting PL '+watts+' W…';
 try{
  const r=await fetch('/api/gpu/power-limit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({gpu:0,watts:watts})});
  const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  vllmMsg.textContent='GPU PL set to '+d.current_w.toFixed(0)+' W';
  await refreshGpuPowerLimitInfo();
  await refreshVllm();
 }catch(e){vllmMsg.textContent='PL error: '+e.message;}
}

async function startVllm(){
 await loadVllmModels();
 const model=vllmModel.value;if(!model){vllmMsg.textContent='No vLLM model found';return;}
 const gpu_ids=selectedVllmGpuIds();if(!gpu_ids.length){vllmMsg.textContent='Select at least one GPU';return;}
 const payload={model:model,port:Number(vllmPort.value),max_model_len:Number(vllmCtx.value),gpu_memory_utilization:Number(vllmMem.value),enforce_eager:vllmEager.checked,dtype:'half',gpu_ids:gpu_ids};
 vllmMsg.textContent='Starting vLLM…';
 try{
  const r=await fetch('/api/vllm/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  vllmMsg.textContent='Started PID '+d.pid+'; model is loading…';setTimeout(refreshVllm,1200);
 }catch(e){vllmMsg.textContent='Start error: '+e.message;}
}

async function applyRestartVllm(){
 await loadVllmModels();
 const model=vllmModel.value;if(!model){vllmMsg.textContent='No vLLM model selected';return;}
 const gpu_ids=selectedVllmGpuIds();if(!gpu_ids.length){vllmMsg.textContent='Select at least one GPU';return;}
 const payload={model:model,port:Number(vllmPort.value),max_model_len:Number(vllmCtx.value),gpu_memory_utilization:Number(vllmMem.value),enforce_eager:vllmEager.checked,dtype:'half',gpu_ids:gpu_ids};
 if(!confirm('Restart vLLM and load '+vllmModel.options[vllmModel.selectedIndex].text+'?'))return;
 vllmMsg.textContent='Stopping current vLLM…';
 try{
  let r=await fetch('/api/vllm/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'stop'})});
  let d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  vllmMsg.textContent='Starting selected model…';
  r=await fetch('/api/vllm/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  vllmMsg.textContent='Started PID '+d.pid+'; loading '+vllmModel.options[vllmModel.selectedIndex].text+'…';
  setTimeout(refreshVllm,1200);
 }catch(e){vllmMsg.textContent='Apply/restart error: '+e.message;}
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
 if(!rows||!rows.length){body.innerHTML='<tr><td colspan="12" class="muted">No UI benchmark runs yet</td></tr>';return;}
 body.innerHTML=[...rows].reverse().map(x=>{
   const mq=(x.model||'—')+' / '+(x.quantization||'—');
   const ttft=(x.ttft_avg_s??0).toFixed(3)+' / '+(x.ttft_max_s??0).toFixed(3)+' s';
   const promptRate=(x.prompt_tok_s_approx??0).toFixed(1);
   const rawId=x.run_id||'';
   const runId=encodeURIComponent(rawId);
   const actions=x.run_id?'<span class="run-report-actions"><a class="run-dl" title="Download this run report" href="/api/vllm/report/run/'+runId+'?download=1">↓</a><a class="run-open" title="Open this run report" target="_blank" href="/api/vllm/report/run/'+runId+'">↗</a></span>':'';
   const checked=x.selected?' checked':'';
   const uploaded=x.git_uploaded?' title="Already uploaded to Git"':'';
   const comment=(x.comment||'').replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
   const sel=x.run_id?'<input class="bench-select vllm-bench-select" type="checkbox" data-run-id="'+runId+'" onchange="saveBenchMeta(\''+runId+'\',{selected:this.checked})"'+checked+uploaded+'>':'—';
   const note=x.run_id?'<input class="bench-comment" value="'+comment+'" placeholder="comment…" onblur="saveBenchMeta(\''+runId+'\',{comment:this.value})">':'';
   const ph=(x.prompt_set_hash||'—'), phShort=ph==='—'?ph:ph.slice(0,12);
   return '<tr><td>'+sel+'</td><td>'+x.concurrency+'</td><td>'+promptProfileBadge(x.prompt_profile)+'</td><td title="'+escHtml(ph)+'">'+escHtml(phShort)+'</td><td>'+(x.prompt_tokens??'—')+'</td><td>'+x.max_tokens+'</td><td>'+x.total_tokens+'</td><td>'+ttft+'</td><td>'+promptRate+'</td><td>'+x.wall_s.toFixed(3)+' s</td><td><b>'+x.aggregate_tok_s.toFixed(2)+' tok/s</b></td><td>'+x.per_request_min_tok_s.toFixed(2)+'–'+x.per_request_max_tok_s.toFixed(2)+' tok/s</td><td>'+mq+actions+'</td><td>'+note+'</td></tr>';
 }).join('');
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
  vllmPid.textContent=s.pid??'—';
  const runningModel=(s.model||'—').split('/').pop();
  vllmLiveModel.textContent=runningModel;
  vllmLiveModel.title=s.model||'';
  vllmUptime.textContent=s.running?fmtUptime(s.uptime_s):'—';
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
    const cpuPct=h.cpu.usage_pct;
    vllmCpu.textContent=cpuPct.toFixed(1)+'%';
    vllmCpu.className=cpuPct>=90?'bad':cpuPct>=70?'warn':'ok';
    vllmCpuSub.textContent='load '+h.cpu.load_1m.toFixed(2)+' • '+(h.cpu.temperature_c==null?'?':h.cpu.temperature_c.toFixed(0)+'°C');
    const ramPct=h.memory.usage_pct;
    vllmRam.textContent=ramPct.toFixed(1)+'%';
    vllmRam.className=ramPct>=90?'bad':ramPct>=75?'warn':'ok';
    vllmRamSub.textContent=(h.memory.used_bytes/1073741824).toFixed(2)+' / '+(h.memory.total_bytes/1073741824).toFixed(2)+' GiB';
    const devices=(h.gpu.devices||[]);
    renderGpuSelector(devices,s.gpu_ids||[0]);
    renderVllmGpuGrid(devices,s.gpu_ids||[0]);
    const selected=(s.gpu_ids||[0]).map(i=>devices[i]).filter(Boolean);
    const totalPl=selected.reduce((a,g)=>a+(Number(g.power_limit_w)||0),0);
    benchPl.textContent=selected.length>1?totalPl.toFixed(0)+' W total':((selected[0]&&selected[0].power_limit_w)??'?')+' W';
    benchGpu.textContent=(s.gpu_ids||[0]).map(i=>'#'+i+' '+((devices[i]&&devices[i].name)||'GPU')).join(' + ');
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

async function saveBenchMeta(runId,patch){
 try{
  const payload={run_id:decodeURIComponent(runId),...patch};
  const r=await fetch('/api/vllm/benchmark/meta',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
 }catch(e){vllmGitMsg.textContent='Save error: '+e.message;}
}

function setAllBenchSelection(value){
 document.querySelectorAll('.bench-select').forEach(cb=>{
  if(cb.checked!==value){cb.checked=value;saveBenchMeta(cb.dataset.runId,{selected:value});}
 });
}

async function deleteSelectedBenchmarks(scope){
 const selector=scope==='history'?'.history-select:checked':'.vllm-bench-select:checked';
 const boxes=[...document.querySelectorAll(selector)];
 if(!boxes.length){
   const msg=scope==='history'?benchHistoryMsg:vllmGitMsg;
   msg.textContent='Select at least one row to delete';
   return;
 }
 if(!confirm('Delete '+boxes.length+' selected benchmark result'+(boxes.length===1?'':'s')+' permanently?'))return;
 const ids=boxes.map(cb=>decodeURIComponent(cb.dataset.runId));
 try{
   const r=await fetch('/api/vllm/benchmark/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run_ids:ids})});
   const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
   if(document.getElementById('vllmGitMsg'))vllmGitMsg.textContent='Deleted '+d.deleted+' result'+(d.deleted===1?'':'s');
   if(document.getElementById('benchHistoryMsg'))benchHistoryMsg.textContent='Deleted '+d.deleted+' result'+(d.deleted===1?'':'s');
   await refreshVllm();
   if(document.getElementById('benchsTab')&&document.getElementById('benchsTab').style.display!=='none')await loadBenchHistory();
 }catch(e){
   const msg=scope==='history'?benchHistoryMsg:vllmGitMsg;
   msg.textContent='Delete error: '+e.message;
 }
}

async function uploadSelectedBenchmarks(){
 const count=[...document.querySelectorAll('.bench-select:checked')].length;
 if(!count){vllmGitMsg.textContent='Select at least one result';return;}
 if(!confirm('Upload '+count+' selected benchmark report'+(count===1?'':'s')+' to Git?'))return;
 vllmGitMsg.textContent='Uploading '+count+' selected report'+(count===1?'':'s')+'…';
 try{
  const r=await fetch('/api/vllm/git/upload-selected',{method:'POST'});
  const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  vllmGitMsg.textContent='Git: '+d.message;
  if(document.getElementById('benchHistoryMsg'))benchHistoryMsg.textContent='Git: '+d.message;
  await refreshVllm();
  if(document.getElementById('benchsTab')&&document.getElementById('benchsTab').style.display!=='none')await loadBenchHistory();
 }catch(e){vllmGitMsg.textContent='Git upload error: '+e.message;}
}

let benchHistoryData=[];

function promptProfileBadge(profile){
 const p=String(profile||'').toLowerCase();
 if(p==='short')return '<span class="prompt-badge s" title="Short prompt">S</span>';
 if(p==='medium')return '<span class="prompt-badge m" title="Medium prompt">M</span>';
 if(p==='long')return '<span class="prompt-badge l" title="Long prompt">L</span>';
 return '<span class="prompt-badge" title="Unknown prompt profile">?</span>';
}

function escHtml(v){
 return String(v??'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

async function loadBenchHistory(){
 benchHistoryMsg.textContent='Loading…';
 try{
  const r=await fetch('/api/vllm/benchmarks');const d=await r.json();
  if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  benchHistoryData=d.benchmarks||[];
  benchStorePath.textContent=d.store||'';
  benchHistoryCount.textContent=benchHistoryData.length+' runs';
  benchHistoryMsg.textContent='';
  populateBenchFilters();
  renderBenchHistory();
 }catch(e){benchHistoryMsg.textContent='History error: '+e.message;}
}

function setFilterOptions(id,values){
 const el=document.getElementById(id);
 const previous=el.value;
 const unique=[...new Set(values.filter(v=>v!==null&&v!==undefined&&String(v)!==''))];
 unique.sort((a,b)=>String(a).localeCompare(String(b),undefined,{numeric:true}));
 el.innerHTML='<option value="">all</option>'+unique.map(v=>'<option value="'+escHtml(v)+'">'+escHtml(v)+'</option>').join('');
 if(unique.map(String).includes(previous))el.value=previous;
}

function _benchGpuLabel(x){
 const ds=x.gpu_devices||[];
 if(ds.length)return ds.map(g=>'#'+g.index+' '+(g.name||'GPU')).join(' + ');
 const ids=x.gpu_ids||[0];
 return ids.map(i=>'#'+i).join(' + ');
}

function populateBenchFilters(){
 setFilterOptions('bfDate',benchHistoryData.map(x=>(x.timestamp||'').slice(0,10)));
 setFilterOptions('bfModel',benchHistoryData.map(x=>x.model||''));
 setFilterOptions('bfQuant',benchHistoryData.map(x=>x.quantization||''));
 setFilterOptions('bfConc',benchHistoryData.map(x=>String(x.concurrency??'')));
 setFilterOptions('bfOut',benchHistoryData.map(x=>String(x.max_tokens??'')));
 setFilterOptions('bfPromptProfile',benchHistoryData.map(x=>x.prompt_profile||''));
 setFilterOptions('bfTemp',benchHistoryData.map(x=>x.temperature==null?'':String(x.temperature)));
 setFilterOptions('bfPcie',benchHistoryData.map(x=>(x.pcie||[]).map(p=>'GPU'+p.index+': G'+(p.gen_current??'?')+' x'+(p.width_current??'?')).join(' + ')));
 setFilterOptions('bfGpus',benchHistoryData.map(x=>_benchGpuLabel(x)));
 setFilterOptions('bfPl',benchHistoryData.map(x=>x.gpu_power_limit_w==null?'':String(Math.round(x.gpu_power_limit_w))));
}

function benchFilters(){
 return {
  date:(bfDate.value||'').trim().toLowerCase(),
  model:(bfModel.value||'').trim().toLowerCase(),
  quant:(bfQuant.value||'').trim().toLowerCase(),
  conc:(bfConc.value||'').trim(),
  out:(bfOut.value||'').trim(),
  prompt_profile:(bfPromptProfile.value||'').trim().toLowerCase(),
  temp:(bfTemp.value||'').trim(),
  thinking:(bfThinking.value||'').trim(),
  pcie:(bfPcie.value||'').trim().toLowerCase(),
  gpus:(bfGpus.value||'').trim().toLowerCase(),
  pl:(bfPl.value||'').trim(),
  git:bfGit.value,
  comment:(bfComment.value||'').trim().toLowerCase(),
  agg:Number(bfAgg.value||0),
 };
}

function filteredBenchHistory(){
 const f=benchFilters();
 return benchHistoryData.filter(x=>{
   const date=(x.timestamp||'').toLowerCase();
   const model=(x.model||'').toLowerCase();
   const quant=(x.quantization||'').toLowerCase();
   const comment=(x.comment||'').toLowerCase();
   if(f.date&&!date.startsWith(f.date))return false;
   if(f.model&&model!==f.model)return false;
   if(f.quant&&quant!==f.quant)return false;
   if(f.conc&&String(x.concurrency)!==f.conc)return false;
   const pcieLabel=(x.pcie||[]).map(p=>'GPU'+p.index+': G'+(p.gen_current??'?')+' x'+(p.width_current??'?')).join(' + ').toLowerCase();
   if(f.pcie&&pcieLabel!==f.pcie)return false;
   if(f.gpus&&_benchGpuLabel(x).toLowerCase()!==f.gpus)return false;
   if(f.pl&&String(Math.round(x.gpu_power_limit_w||0))!==f.pl)return false;
   if(f.git==='yes'&&!x.git_uploaded)return false;
   if(f.git==='no'&&x.git_uploaded)return false;
   if(f.comment&&!comment.includes(f.comment))return false;
   if(f.agg&&Number(x.aggregate_tok_s||0)<f.agg)return false;
   return true;
 });
}

function benchPeerKey(x){
 return [
  x.model||'',x.quantization||'',x.prompt_profile||'',
  x.max_tokens||'',x.enable_thinking?'1':'0',
  _benchGpuLabel(x)
 ].join('|');
}

function normalized(v,min,max,invert=false){
 v=Number(v||0);min=Number(min||0);max=Number(max||0);
 if(max<=min)return 1;
 let n=(v-min)/(max-min);
 if(invert)n=1-n;
 return Math.max(0,Math.min(1,n));
}

function benchScores(rows){
 const groups={};
 rows.forEach(x=>{const k=benchPeerKey(x);(groups[k]??=[]).push(x);});
 const out={};
 Object.values(groups).forEach(group=>{
   const vals=k=>group.map(x=>Number(x[k]||0)).filter(Number.isFinite);
   const agg=vals('aggregate_tok_s'), per=vals('per_request_min_tok_s'), ttft=vals('ttft_avg_s');
   const eff=group.map(x=>Number(x.gpu_power_avg_w)>0?Number(x.aggregate_tok_s||0)/Number(x.gpu_power_avg_w):0);
   const range=a=>[Math.min(...a),Math.max(...a)];
   const [amin,amax]=range(agg),[pmin,pmax]=range(per),[tmin,tmax]=range(ttft),[emin,emax]=range(eff);
   group.forEach((x,i)=>{
     const score=100*(
       0.35*normalized(x.aggregate_tok_s,amin,amax)+
       0.25*normalized(x.per_request_min_tok_s,pmin,pmax)+
       0.20*normalized(eff[i],emin,emax)+
       0.20*normalized(x.ttft_avg_s,tmin,tmax,true)
     );
     out[x.run_id]=Math.round(score);
   });
 });
 return out;
}

function benchRecommendation(x){
 const tags=[];
 const profile=String(x.prompt_profile||'').toLowerCase();
 const per=Number(x.per_request_min_tok_s||0);
 const ttft=Number(x.ttft_avg_s||0);
 const conc=Number(x.concurrency||1);

 if(profile==='short')tags.push(['coding','Light coding']);
 else if(profile==='medium')tags.push(['coding','Coding / chat']);
 else if(profile==='long')tags.push(['longctx','Repo / agent']);

 if(ttft<=0.5&&per>=40)tags.push(['interactive','Interactive']);
 else if(ttft<=2&&per>=20)tags.push(['interactive','Usable live']);
 else if(ttft>5)tags.push(['batch','Offline']);

 if(conc>=8)tags.push(['batch','Batch / API']);
 else if(conc>=4&&Number(x.aggregate_tok_s||0)>=150)tags.push(['batch','Multi-user']);

 return tags.slice(0,3);
}

function ratingHtml(score){
 const cls=score>=80?'r-high':score>=55?'r-mid':'r-low';
 return '<span class="rating-pill '+cls+'" title="Balanced score within comparable workload peers">'+score+'/100</span>';
}

function recHtml(x){
 return '<div class="rec-badges">'+benchRecommendation(x).map(t=>'<span class="rec-tag '+t[0]+'">'+escHtml(t[1])+'</span>').join('')+'</div>';
}

function updateBenchBestCards(rows,scores){
 if(!rows.length){benchBestAgg.textContent=benchBestEff.textContent=benchBestScore.textContent='—';return;}
 const bestAgg=rows.reduce((a,b)=>Number(b.aggregate_tok_s||0)>Number(a.aggregate_tok_s||0)?b:a,rows[0]);
 const eff=x=>Number(x.gpu_power_avg_w)>0?Number(x.aggregate_tok_s||0)/Number(x.gpu_power_avg_w):0;
 const bestEff=rows.reduce((a,b)=>eff(b)>eff(a)?b:a,rows[0]);
 const bestScore=rows.reduce((a,b)=>(scores[b.run_id]||0)>(scores[a.run_id]||0)?b:a,rows[0]);
 benchBestAgg.textContent=Number(bestAgg.aggregate_tok_s||0).toFixed(2)+' tok/s';
 benchBestAggSub.textContent=(bestAgg.model||'—')+' • '+(bestAgg.prompt_profile||'—')+' • conc '+bestAgg.concurrency+' • PL '+Math.round(bestAgg.gpu_power_limit_w||0)+' W';
 benchBestEff.textContent=eff(bestEff).toFixed(3)+' tok/s/W';
 benchBestEffSub.textContent=(bestEff.model||'—')+' • '+(bestEff.prompt_profile||'—')+' • conc '+bestEff.concurrency+' • PL '+Math.round(bestEff.gpu_power_limit_w||0)+' W';
 benchBestScore.textContent=(scores[bestScore.run_id]||0)+'/100';
 benchBestScoreSub.textContent=(bestScore.model||'—')+' • '+(bestScore.prompt_profile||'—')+' • conc '+bestScore.concurrency+' • balanced';
}

function renderBenchHistory(){
 const body=document.getElementById('benchHistoryRows');
 if(!body)return;
 const rows=filteredBenchHistory().slice().reverse();
 const scores=benchScores(rows);
 updateBenchBestCards(rows,scores);
 benchHistoryCount.textContent=rows.length+' shown / '+benchHistoryData.length+' total';
 if(!rows.length){body.innerHTML='<tr><td colspan="23" class="muted">No matching benchmark rows</td></tr>';return;}
 const maxAgg=Math.max(...rows.map(x=>Number(x.aggregate_tok_s||0)));
 const maxEff=Math.max(...rows.map(x=>Number(x.gpu_power_avg_w)>0?Number(x.aggregate_tok_s||0)/Number(x.gpu_power_avg_w):0));
 body.innerHTML=rows.map(x=>{
   const rawId=x.run_id||'', runId=encodeURIComponent(rawId);
   const checked=x.selected?' checked':'';
   const sel=rawId?'<input class="bench-select history-select" type="checkbox" data-run-id="'+runId+'" onchange="saveBenchMeta(\''+runId+'\',{selected:this.checked})"'+checked+'>':'—';
   const mq=escHtml((x.model||'—')+' / '+(x.quantization||'—'));
   const date=escHtml((x.timestamp||'—').replace('T',' ').slice(0,19));
   const ttft=(x.ttft_avg_s??0).toFixed(3)+' / '+(x.ttft_max_s??0).toFixed(3);
   const per=(x.per_request_min_tok_s??0).toFixed(2)+'–'+(x.per_request_max_tok_s??0).toFixed(2);
   const pl=x.gpu_power_limit_w==null?'—':Number(x.gpu_power_limit_w).toFixed(0)+' W';
   const pwr=(x.gpu_power_avg_w==null?'—':Number(x.gpu_power_avg_w).toFixed(1))+' / '+(x.gpu_power_peak_w==null?'—':Number(x.gpu_power_peak_w).toFixed(1));
   const temp=x.gpu_temp_peak_c==null?'—':Number(x.gpu_temp_peak_c).toFixed(0)+'°C';
   const git=x.git_uploaded?'<span class="git-state-ok">✓ Git</span>':'<span class="git-state-no">—</span>';
   const actions=rawId?'<span class="run-report-actions"><a class="run-dl" title="Download report" href="/api/vllm/report/run/'+runId+'?download=1">↓</a><a class="run-open" title="Open report" target="_blank" href="/api/vllm/report/run/'+runId+'">↗</a></span>':'—';
   const note=rawId?'<input class="bench-comment" value="'+escHtml(x.comment||'')+'" placeholder="comment…" onblur="saveHistoryComment(\''+runId+'\',this.value)">':'';
   const pcie=(x.pcie||[]).map(p=>'GPU'+p.index+': G'+(p.gen_current??'?')+' x'+(p.width_current??'?')).join(' + ')||'—';
   const ph=(x.prompt_set_hash||'—'); const phShort=ph==='—'?ph:ph.slice(0,12);
   const agg=Number(x.aggregate_tok_s||0);
   const efficiency=Number(x.gpu_power_avg_w)>0?agg/Number(x.gpu_power_avg_w):0;
   const ttftClass=Number(x.ttft_avg_s||0)<=0.5?'metric-good':Number(x.ttft_avg_s||0)<=2?'metric-warn':'metric-bad';
   const tempClass=Number(x.gpu_temp_peak_c||0)<65?'metric-good':Number(x.gpu_temp_peak_c||0)<78?'metric-warn':'metric-bad';
   const rowClass=(agg===maxAgg?' row-best-agg':'')+(Math.abs(efficiency-maxEff)<0.000001?' row-best-eff':'');
   return '<tr class="'+rowClass.trim()+'"><td>'+sel+'</td><td>'+date+'</td><td>'+mq+'</td><td>'+escHtml(_benchGpuLabel(x))+'</td><td>'+escHtml(pcie)+'</td><td>'+(x.enable_thinking?'ON':'OFF')+'</td><td>'+promptProfileBadge(x.prompt_profile)+'</td><td title="'+escHtml(ph)+'">'+escHtml(phShort)+'</td><td>'+x.concurrency+'</td><td>'+(x.prompt_tokens??'—')+'</td><td>'+x.max_tokens+'</td><td class="'+ttftClass+'">'+ttft+'</td><td>'+Number(x.wall_s||0).toFixed(3)+' s</td><td class="'+(agg===maxAgg?'metric-good':'')+'"><b>'+agg.toFixed(2)+'</b></td><td>'+per+'</td><td>'+pl+'</td><td>'+pwr+' W</td><td class="'+tempClass+'">'+temp+'</td><td>'+ratingHtml(scores[x.run_id]||0)+'</td><td>'+recHtml(x)+'</td><td>'+git+'</td><td>'+actions+'</td><td>'+note+'</td></tr>';
 }).join('');
}

async function saveHistoryComment(runId,value){
 await saveBenchMeta(runId,{comment:value});
 const raw=decodeURIComponent(runId);
 const row=benchHistoryData.find(x=>x.run_id===raw);
 if(row)row.comment=value;
}

function setAllHistorySelection(value){
 document.querySelectorAll('.history-select').forEach(cb=>{
  if(cb.checked!==value){cb.checked=value;saveBenchMeta(cb.dataset.runId,{selected:value});}
 });
 benchHistoryData.forEach(x=>{
  const visible=[...document.querySelectorAll('.history-select')].some(cb=>decodeURIComponent(cb.dataset.runId)===x.run_id);
  if(visible)x.selected=value;
 });
}

function clearBenchFilters(){
 ['bfDate','bfModel','bfQuant','bfConc','bfOut','bfPromptProfile','bfTemp','bfThinking','bfPcie','bfGpus','bfPl','bfComment','bfAgg'].forEach(id=>document.getElementById(id).value='');
 bfGit.value='';
 renderBenchHistory();
}

function syncBenchThinkingBadge(){const e=document.getElementById('benchThinkingState');if(e)e.textContent=benchThinking.value==='on'?'ON':'OFF';}
async function runVllmBench(concurrency){
 const out=Number(benchOutTokens.value);
 const temperature=Number(benchTemperature.value);
 const prompt_profile=benchPromptProfile.value;
 const enable_thinking=benchThinking.value==='on';syncBenchThinkingBadge();
 if(!Number.isInteger(concurrency)||concurrency<1||concurrency>128){vllmBenchMsg.textContent='Concurrency must be 1..128';return;}
 vllmBenchMsg.textContent='Running '+concurrency+' × '+out+' • '+prompt_profile+' • thinking '+(enable_thinking?'ON':'OFF')+'…';
 try{
  const payload={concurrency:concurrency,max_tokens:out,temperature:temperature,prompt_profile:prompt_profile,enable_thinking:enable_thinking};
  const r=await fetch('/api/vllm/benchmark',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  vllmBenchMsg.textContent=concurrency+' × '+out+' = '+d.result.aggregate_tok_s.toFixed(2)+' tok/s • TTFT '+d.result.ttft_avg_s.toFixed(3)+' s';
  await refreshVllm();
 }catch(e){vllmBenchMsg.textContent='Benchmark error: '+e.message;}
}
setTimeout(()=>loadVllmModels(true),300);
setInterval(()=>{if(document.getElementById('vllmTab')&&document.getElementById('vllmTab').style.display!=='none')refreshVllm();},2000);
setInterval(()=>{if(document.getElementById('vllmTab')&&document.getElementById('vllmTab').style.display!=='none')refreshGpuPowerLimitInfo();},10000);
'''
    dashboard = dashboard.replace(
        "refresh();setInterval(refresh,2000);",
        js + "\nrefresh();setInterval(refresh,2000);",
        1,
    )

    base.DASHBOARD = dashboard
    dynamic.base.DASHBOARD = dashboard
    dynamic.app = base.app

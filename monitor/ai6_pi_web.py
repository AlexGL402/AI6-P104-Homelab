#!/usr/bin/env python3
"""Web UI/API for the local Pi coding agent."""

import json
import os
import subprocess
import time
import urllib.request
import urllib.parse
from pathlib import Path

from pydantic import BaseModel, Field
import ai6_monitor_dynamic as dynamic

base = dynamic.base

REPO = Path.home() / "AI6-P104-Homelab"
PI = Path("/usr/bin/pi")
PROVIDERS_FILE = Path.home() / ".local/state/ai6-monitor/pi-providers.json"
BENCH_FILE = Path.home() / ".local/state/ai6-monitor/pi-benchmarks.jsonl"
NODES_FILE = Path.home() / ".local/state/ai6-monitor/pi-nodes.json"
MODELS = {
    "8b": ("ai6-ollama", "qwen3:8b", "Qwen3 8B"),
    "14b": ("ai6-ollama", "qwen3:14b", "Qwen3 14B"),
}


class ProviderCommand(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    base_url: str = Field(min_length=8, max_length=256)
    api_key: str = Field(default="ollama", max_length=256)


class NodeCommand(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    monitor_url: str = Field(min_length=8, max_length=256)
    disabled_gpus: list[int] = Field(default_factory=list)


class NodeStartCommand(BaseModel):
    name: str
    model: str
    gpus: list[int] = Field(default_factory=list)
    split: str = "layer"
    ctx: int = 8192
    ngl: int = 999
    alias: str = "pi-remote"


class NodeStopCommand(BaseModel):
    name: str
    port: int


class NodeGpuPolicyCommand(BaseModel):
    name: str
    gpu: int
    allow_auto: bool


class BenchRunCommand(BaseModel):
    name: str
    backend: str = "auto"
    model: str
    gpus: list[int] = Field(default_factory=list)
    ctx: int = 4096
    tokens: int = 256
    split: str = "layer"


class PiChatCommand(BaseModel):
    prompt: str = Field(min_length=1, max_length=12000)
    model: str = Field(default="8b", max_length=256)
    history: list[dict] = Field(default_factory=list)


def _load_nodes():
    try:
        return json.loads(NODES_FILE.read_text())
    except Exception:
        return []


def _save_nodes(items):
    NODES_FILE.parent.mkdir(parents=True, exist_ok=True)
    NODES_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n")


def _json_request(url, method="GET", payload=None, timeout=8):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _node(name):
    for item in _load_nodes():
        if item.get("name") == name:
            return item
    raise base.HTTPException(status_code=404, detail="managed node not found")


def _provider_key(name):
    return "ai6-web-" + "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")


def _load_providers():
    try:
        return json.loads(PROVIDERS_FILE.read_text())
    except Exception:
        return []


def _save_providers(items):
    PROVIDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROVIDERS_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n")


def _discover_models(base_url, api_key="ollama"):
    req = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": "Bearer " + (api_key or "ollama")},
    )
    with urllib.request.urlopen(req, timeout=6) as response:
        data = json.loads(response.read().decode("utf-8"))
    return [str(x.get("id")) for x in data.get("data", []) if x.get("id")]


def _sync_pi_models():
    path = Path.home() / ".pi/agent/models.json"
    try:
        data = json.loads(path.read_text())
    except Exception:
        data = {}
    for item in _load_providers():
        key = _provider_key(item["name"])
        data.setdefault("providers", {})[key] = {
            "api": "openai-completions",
            "baseUrl": item["base_url"].rstrip("/"),
            "apiKey": item.get("api_key") or "ollama",
            "models": [
                {"id": mid, "name": mid + " @ " + item["name"], "contextWindow": 16384,
                 "maxTokens": 4096, "reasoning": False}
                for mid in item.get("models", [])
            ],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _append_benchmark(row):
    BENCH_FILE.parent.mkdir(parents=True, exist_ok=True)
    with BENCH_FILE.open("a") as fp:
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_benchmarks(limit=30):
    try:
        rows = [json.loads(x) for x in BENCH_FILE.read_text().splitlines() if x.strip()]
        return rows[-limit:][::-1]
    except Exception:
        return []


def _history_prompt(history, prompt):
    rows = []
    for item in history[-8:]:
        role = str(item.get("role") or "")[:16]
        content = str(item.get("content") or "")[:5000]
        if role in ("user", "assistant") and content:
            rows.append(f"{role.upper()}: {content}")
    if not rows:
        return prompt
    return (
        "Continue this coding-agent conversation. The previous messages are context only.\n\n"
        + "\n\n".join(rows)
        + "\n\nUSER: " + prompt
    )


@base.app.get("/api/pi/status")
def pi_status():
    models = []
    try:
        p = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=4,
        )
        text = p.stdout or ""
        for key, (_, model_id, label) in MODELS.items():
            models.append({"key": key, "id": model_id, "label": label, "installed": model_id in text})
    except Exception:
        for key, (_, model_id, label) in MODELS.items():
            models.append({"key": key, "id": model_id, "label": label, "installed": False})
    return {
        "pi": str(PI),
        "pi_installed": PI.is_file(),
        "repo": str(REPO),
        "models": models,
        "terminal_port": 8093,
    }


@base.app.get("/api/pi/nodes")
def pi_nodes():
    out = []
    for item in _load_nodes():
        row = dict(item)
        try:
            root = item["monitor_url"].rstrip("/")
            stats = _json_request(root + "/api/stats", timeout=3)
            models = _json_request(root + "/api/models", timeout=3).get("models", [])
            ollama_models = []
            try:
                parsed = urllib.parse.urlsplit(root)
                ollama_url = f"{parsed.scheme}://{parsed.hostname}:11434/v1"
                ollama_models = _discover_models(ollama_url, "ollama")
            except Exception:
                pass
            workers = stats.get("llama", {}).get("workers", {})
            # Bench/Managed Node model discovery must also include models that are
            # already running. A worker can use a GGUF outside /api/models' scan
            # path, so relying on /api/models alone leaves the selector empty.
            known_paths = {str(m.get("path")) for m in models if m.get("path")}
            running_models = []
            for port, worker in workers.items():
                mid = str((worker or {}).get("model") or "").strip()
                if not mid:
                    continue
                running_models.append({"label": f"running:{port} • {Path(mid).name}", "path": mid})
                if mid not in known_paths and (mid.endswith(".gguf") or mid.startswith("/")):
                    models.append({"label": f"running:{port} • {Path(mid).name}", "path": mid})
                    known_paths.add(mid)
            row.update({"online": True, "host": stats.get("host", {}),
                        "gpus": stats.get("gpu", {}).get("devices", []),
                        "workers": workers,
                        "models": models, "running_models": running_models,
                        "ollama_models": ollama_models})
        except Exception as e:
            row.update({"online": False, "error": str(e), "gpus": [], "workers": {}, "models": []})
        out.append(row)
    return {"nodes": out}


@base.app.post("/api/pi/nodes")
def pi_node_add(cmd: NodeCommand):
    root = cmd.monitor_url.rstrip("/")
    try:
        stats = _json_request(root + "/api/stats", timeout=4)
        models = _json_request(root + "/api/models", timeout=4).get("models", [])
    except Exception as e:
        raise base.HTTPException(status_code=409, detail=f"node test failed: {e}")
    items = [x for x in _load_nodes() if x.get("name") != cmd.name]
    old = next((x for x in _load_nodes() if x.get("name") == cmd.name), {})
    items.append({"name": cmd.name, "monitor_url": root,
                  "disabled_gpus": cmd.disabled_gpus if cmd.disabled_gpus else old.get("disabled_gpus", [])})
    _save_nodes(items)
    return {"ok": True, "host": stats.get("host", {}), "models": len(models),
            "gpus": len(stats.get("gpu", {}).get("devices", []))}


@base.app.delete("/api/pi/nodes/{name}")
def pi_node_delete(name: str):
    _save_nodes([x for x in _load_nodes() if x.get("name") != name])
    return {"ok": True}


@base.app.post("/api/pi/nodes/gpu-policy")
def pi_node_gpu_policy(cmd: NodeGpuPolicyCommand):
    items = _load_nodes()
    found = False
    for item in items:
        if item.get("name") != cmd.name:
            continue
        found = True
        disabled = set(int(x) for x in item.get("disabled_gpus", []))
        if cmd.allow_auto:
            disabled.discard(cmd.gpu)
        else:
            disabled.add(cmd.gpu)
        item["disabled_gpus"] = sorted(disabled)
    if not found:
        raise base.HTTPException(status_code=404, detail="managed node not found")
    _save_nodes(items)
    return {"ok": True}


@base.app.post("/api/pi/nodes/start")
def pi_node_start(cmd: NodeStartCommand):
    item = _node(cmd.name)
    root = item["monitor_url"].rstrip("/")
    gpus = list(cmd.gpus)
    if not gpus:
        try:
            stats = _json_request(root + "/api/stats", timeout=4)
            disabled = set(int(x) for x in item.get("disabled_gpus", []))
            devices = stats.get("gpu", {}).get("devices", [])
            candidates = []
            for g in devices:
                idx = int(g.get("index", -1))
                if idx < 0 or idx in disabled:
                    continue
                used = float(g.get("memory_used_mib") or 0)
                total = float(g.get("memory_total_mib") or 0)
                util = float(g.get("util_gpu_pct") or 0)
                candidates.append((used / total if total else 1.0, util, idx))
            if not candidates:
                raise RuntimeError("no enabled GPUs available")
            candidates.sort()
            gpus = [candidates[0][2]]
        except Exception as e:
            raise base.HTTPException(status_code=409, detail=f"auto GPU selection failed: {e}")
    try:
        started = _json_request(root + "/api/workers/custom/start", "POST", {
            "model": cmd.model, "gpus": gpus, "split": cmd.split,
            "ctx": cmd.ctx, "ngl": cmd.ngl, "port": None, "alias": cmd.alias,
        }, timeout=12)
    except Exception as e:
        raise base.HTTPException(status_code=409, detail=f"remote start failed: {e}")
    port = int(started["port"])
    parsed = urllib.parse.urlsplit(root)
    worker_url = f"{parsed.scheme}://{parsed.hostname}:{port}/v1"
    ready = False
    last_error = ""
    for _ in range(90):
        try:
            models = _discover_models(worker_url, "ollama")
            if models:
                ready = True
                break
        except Exception as e:
            last_error = str(e)
        time.sleep(1)
    if not ready:
        return {"ok": True, "started": True, "ready": False, "port": port,
                "worker_url": worker_url, "detail": last_error or "model still loading"}
    provider_name = f"{cmd.name}-{cmd.alias}-{port}"
    items = [x for x in _load_providers()
             if x.get("base_url", "").rstrip("/") != worker_url.rstrip("/")]
    items.append({"name": provider_name, "base_url": worker_url,
                  "api_key": "ollama", "models": models, "managed_node": cmd.name,
                  "managed_port": port})
    _save_providers(items)
    _sync_pi_models()
    return {"ok": True, "started": True, "ready": True, "port": port,
            "worker_url": worker_url, "provider": provider_name, "models": models, "gpus": gpus}


@base.app.post("/api/pi/nodes/stop")
def pi_node_stop(cmd: NodeStopCommand):
    item = _node(cmd.name)
    root = item["monitor_url"].rstrip("/")
    try:
        result = _json_request(root + "/api/workers/custom/stop", "POST",
                               {"port": cmd.port}, timeout=10)
    except Exception as e:
        raise base.HTTPException(status_code=409, detail=f"remote stop failed: {e}")
    _save_providers([x for x in _load_providers()
                     if not (x.get("managed_node") == cmd.name and x.get("managed_port") == cmd.port)])
    _sync_pi_models()
    return {"ok": True, "result": result}


@base.app.get("/api/pi/providers")
def pi_providers():
    return {"providers": _load_providers()}


@base.app.post("/api/pi/providers")
def pi_provider_add(cmd: ProviderCommand):
    try:
        models = _discover_models(cmd.base_url, cmd.api_key)
    except Exception as e:
        raise base.HTTPException(status_code=409, detail=f"endpoint test failed: {e}")
    if not models:
        raise base.HTTPException(status_code=409, detail="endpoint returned no models")
    items = [x for x in _load_providers() if x.get("name") != cmd.name]
    items.append({"name": cmd.name, "base_url": cmd.base_url.rstrip("/"),
                  "api_key": cmd.api_key, "models": models})
    _save_providers(items)
    _sync_pi_models()
    return {"ok": True, "name": cmd.name, "models": models}


@base.app.delete("/api/pi/providers/{name}")
def pi_provider_delete(name: str):
    _save_providers([x for x in _load_providers() if x.get("name") != name])
    _sync_pi_models()
    return {"ok": True}


@base.app.get("/api/pi/benchmarks")
def pi_benchmarks():
    return {"runs": _load_benchmarks()}


@base.app.post("/api/pi/benchmark/run")
def pi_benchmark_run(cmd: BenchRunCommand):
    item = _node(cmd.name)
    root = item["monitor_url"].rstrip("/")
    parsed = urllib.parse.urlsplit(root)
    hostroot = f"{parsed.scheme}://{parsed.hostname}"
    backend = cmd.backend
    model = cmd.model
    if backend == "auto":
        backend = "ollama" if model.startswith("ollama:") else "llama.cpp"
    if model.startswith("ollama:"):
        model = model[7:]
    elif model.startswith("gguf:"):
        model = model[5:]
    if backend not in ("ollama", "llama.cpp"):
        raise base.HTTPException(status_code=400, detail="Benchmark Runner currently supports Ollama and llama.cpp")
    if not (2048 <= cmd.ctx <= 131072) or not (16 <= cmd.tokens <= 4096):
        raise base.HTTPException(status_code=400, detail="invalid context/tokens")

    prompt = ("Write a concise Python function that recursively finds the 10 largest files "
              "under a directory, handles permission errors, and explain the complexity.")
    port = None
    used_gpus = list(cmd.gpus)
    started_at = time.monotonic()
    try:
        if backend == "ollama":
            url = hostroot + ":11434/api/generate"
            result = _json_request(url, "POST", {
                "model": model, "prompt": prompt, "stream": False,
                "options": {"temperature": 0, "num_ctx": cmd.ctx, "num_predict": cmd.tokens},
            }, timeout=600)
            prompt_n = int(result.get("prompt_eval_count") or 0)
            eval_n = int(result.get("eval_count") or 0)
            pd = float(result.get("prompt_eval_duration") or 0) / 1e9
            ed = float(result.get("eval_duration") or 0) / 1e9
            prompt_tps = round(prompt_n / pd, 2) if pd else None
            gen_tps = round(eval_n / ed, 2) if ed else None
        else:
            if not used_gpus:
                stats0 = _json_request(root + "/api/stats", timeout=5)
                disabled = set(int(x) for x in item.get("disabled_gpus", []))
                candidates = []
                for g in stats0.get("gpu", {}).get("devices", []):
                    idx = int(g.get("index", -1))
                    if idx < 0 or idx in disabled:
                        continue
                    total = float(g.get("memory_total_mib") or 0)
                    used = float(g.get("memory_used_mib") or 0)
                    candidates.append((used / total if total else 1.0, float(g.get("util_gpu_pct") or 0), idx))
                if not candidates:
                    raise RuntimeError("no enabled GPU available")
                candidates.sort()
                used_gpus = [candidates[0][2]]
            started = _json_request(root + "/api/workers/custom/start", "POST", {
                "model": model, "gpus": used_gpus, "split": cmd.split,
                "ctx": cmd.ctx, "ngl": 999, "port": None, "alias": "bench",
            }, timeout=15)
            port = int(started["port"])
            api = hostroot + f":{port}"
            ready = False
            for _ in range(120):
                try:
                    if _json_request(api + "/health", timeout=2).get("status") == "ok":
                        ready = True
                        break
                except Exception:
                    pass
                time.sleep(1)
            if not ready:
                raise RuntimeError(f"worker {port} did not become ready")
            result = _json_request(api + "/v1/completions", "POST", {
                "model": "bench", "prompt": prompt, "temperature": 0,
                "max_tokens": cmd.tokens,
            }, timeout=600)
            timings = result.get("timings") or {}
            prompt_n = int(timings.get("prompt_n") or (result.get("usage") or {}).get("prompt_tokens") or 0)
            eval_n = int(timings.get("predicted_n") or (result.get("usage") or {}).get("completion_tokens") or 0)
            prompt_tps = timings.get("prompt_per_second")
            gen_tps = timings.get("predicted_per_second")

        stats = _json_request(root + "/api/stats", timeout=5)
        devices = stats.get("gpu", {}).get("devices", [])
        chosen = devices if not used_gpus else [g for g in devices if int(g.get("index", -1)) in used_gpus]
        row = {
            "ts": int(time.time()), "kind": "model-bench", "node": cmd.name,
            "backend": backend, "model": model, "gpus": used_gpus,
            "ctx": cmd.ctx, "tokens": cmd.tokens, "prompt_tokens": prompt_n,
            "output": eval_n, "prompt_tps": round(float(prompt_tps), 2) if prompt_tps is not None else None,
            "gen_tps": round(float(gen_tps), 2) if gen_tps is not None else None,
            "elapsed_s": round(time.monotonic() - started_at, 2),
            "gpu": [{
                "index": g.get("index"), "name": g.get("name"),
                "memory_used_mib": g.get("memory_used_mib"),
                "power_w": g.get("power_w"), "temp_c": g.get("temp_c"),
            } for g in chosen],
        }
        _append_benchmark(row)
        return {"ok": True, "run": row}
    except base.HTTPException:
        raise
    except Exception as e:
        row = {"ts": int(time.time()), "kind": "model-bench", "node": cmd.name,
               "backend": backend, "model": model, "gpus": used_gpus,
               "ctx": cmd.ctx, "tokens": cmd.tokens, "error": str(e),
               "elapsed_s": round(time.monotonic() - started_at, 2)}
        _append_benchmark(row)
        raise base.HTTPException(status_code=409, detail=str(e))
    finally:
        if port is not None:
            try:
                _json_request(root + "/api/workers/custom/stop", "POST", {"port": port}, timeout=10)
            except Exception:
                pass


@base.app.post("/api/pi/chat")
def pi_chat(cmd: PiChatCommand):
    if not PI.is_file():
        raise base.HTTPException(status_code=409, detail=f"Pi not found: {PI}")
    if cmd.model in MODELS:
        provider, model_id, label = MODELS[cmd.model]
    else:
        provider = model_id = label = None
        for item in _load_providers():
            key = _provider_key(item["name"])
            for mid in item.get("models", []):
                if cmd.model == key + "::" + mid:
                    provider, model_id, label = key, mid, mid + " @ " + item["name"]
                    break
        if not provider:
            raise base.HTTPException(status_code=400, detail="unknown Pi model")
    prompt = _history_prompt(cmd.history, cmd.prompt)
    args = [
        str(PI),
        "--mode", "json",
        "--approve",
        "--provider", provider,
        "--model", model_id,
        prompt,
    ]
    env = os.environ.copy()
    env["HOME"] = str(Path.home())
    started = time.monotonic()
    try:
        p = subprocess.run(
            args,
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        raise base.HTTPException(status_code=504, detail="Pi timed out after 10 minutes")
    elapsed = round(time.monotonic() - started, 2)
    if p.returncode != 0:
        err = (p.stderr or p.stdout or f"Pi exited {p.returncode}").strip()
        raise base.HTTPException(status_code=409, detail=err[-6000:])
    response_parts = []
    usage = {}
    tool_calls = 0
    tool_started = {}
    tool_time_s = 0.0
    for line in (p.stdout or "").splitlines():
        try:
            event = json.loads(line)
        except Exception:
            continue
        if isinstance(event.get("usage"), dict):
            usage.update(event["usage"])
        if event.get("type") == "message_end":
            msg = event.get("message") or {}
            if isinstance(msg.get("usage"), dict):
                usage.update(msg["usage"])
            if msg.get("role") == "assistant":
                for part in msg.get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                        response_parts.append(part["text"])
        if event.get("type") == "tool_execution_start":
            tool_calls += 1
            key = str(event.get("toolCallId") or event.get("id") or tool_calls)
            tool_started[key] = time.monotonic()
        if event.get("type") in ("tool_execution_end", "tool_execution_result"):
            key = str(event.get("toolCallId") or event.get("id") or "")
            if key in tool_started:
                tool_time_s += time.monotonic() - tool_started.pop(key)

    response = response_parts[-1] if response_parts else "(no assistant text)"
    inp = usage.get("input") or usage.get("inputTokens") or usage.get("promptTokens") or 0
    out = usage.get("output") or usage.get("outputTokens") or usage.get("completionTokens") or 0
    total = usage.get("totalTokens") or ((inp or 0) + (out or 0))
    agent_tok_s = round(out / elapsed, 2) if out and elapsed else None
    row = {
        "ts": int(time.time()), "label": label, "model": model_id,
        "elapsed_s": elapsed, "tool_time_s": round(tool_time_s, 2),
        "llm_plus_overhead_s": round(max(0.0, elapsed - tool_time_s), 2),
        "input": inp, "output": out, "agent_tok_s": agent_tok_s,
        "tool_calls": tool_calls,
    }
    _append_benchmark(row)
    return {
        "ok": True,
        "model": model_id,
        "label": label,
        "elapsed_s": elapsed,
        "response": response.strip(),
        "usage": {"input": inp, "output": out, "total": total},
        "avg_output_tok_s": agent_tok_s,
        "tool_calls": tool_calls,
        "tool_time_s": row["tool_time_s"],
        "llm_plus_overhead_s": row["llm_plus_overhead_s"],
        "stderr": (p.stderr or "").strip()[-2000:],
    }


def install():
    dashboard = base.DASHBOARD
    dashboard = dashboard.replace(
        '<button id="tabPiBtn" class="tab-btn" onclick="showTopTab(\'pi\')">Pi Agent</button>',
        '<button id="tabPiBtn" class="tab-btn" onclick="showTopTab(\'pi\')">Pi Coder</button>',
        1,
    )

    old = r'''<div id="piTab" style="display:none">
  <div class="section">
    <div class="section-head">
      <div>
        <div class="section-title">Pi Coder Agent</div>
        <div class="section-sub">Pi 0.85.1 • working directory: ~/AI6-P104-Homelab</div>
      </div>
      <button onclick="document.getElementById('piFrame').src=piAgentUrl()">Reconnect</button>
    </div>
    <iframe id="piFrame"
      title="Pi Coder Agent"
      style="width:100%;height:720px;border:1px solid #333;border-radius:10px;background:#000">
    </iframe>
  </div>
</div>'''

    new = r'''<div id="piTab" style="display:none">
  <div class="section pi-web">
    <div class="section-head">
      <div>
        <div class="section-title">Pi Coder Web</div>
        <div class="section-sub">Local coding agent • repo tools enabled • Ollama on CMP40</div>
      </div>
      <div class="pi-toolbar">
        <select id="piModel">
          <option value="8b" selected>Qwen3 8B — fast / default</option>
          <option value="14b">Qwen3 14B — quality / slow</option>
        </select>
        <button onclick="togglePiSettings()">⚙ Workers</button>
        <button onclick="clearPiChat()">New chat</button>
        <button onclick="openPiTerminal()">Terminal</button>
      </div>
    </div>
    <div id="piStatus" class="muted">checking Pi…</div>
    <div id="piSettings" class="pi-settings" style="display:none">
      <b>Managed AI6 nodes</b>
      <div class="pi-settings-row">
        <input id="piNodeName" placeholder="Node: T5600">
        <input id="piNodeUrl" placeholder="http://192.168.1.2:8090">
        <button onclick="addPiNode()">Test + Save node</button>
      </div>
      <div id="piNodes" class="pi-nodes"></div>
      <hr style="border:0;border-top:1px solid #333;margin:12px 0">
      <b>Model workers / OpenAI-compatible endpoints</b>
      <div class="pi-settings-row">
        <input id="piWorkerName" placeholder="Name: T5600 / AI6">
        <input id="piWorkerUrl" placeholder="http://10.36.1.103:8092/v1">
        <input id="piWorkerKey" placeholder="API key (optional)">
        <button onclick="addPiWorker()">Test + Discover + Save</button>
      </div>
      <div id="piWorkers" class="section-sub"></div>
    </div>
    <div id="piChat" class="pi-chat">
      <div class="pi-empty">Pi Coder is ready. Ask it to inspect, edit or test this repository.</div>
    </div>
    <div class="pi-compose">
      <textarea id="piPrompt" rows="4" placeholder="Ask Pi to inspect or change the repository…"></textarea>
      <button id="piSend" onclick="sendPi()">Send</button>
    </div>
    <div class="section-sub">Ctrl+Enter sends • tool access is Pi's normal read/bash/edit/write/grep/find/ls set • Terminal fallback stays on :8093</div>
  </div>
</div>'''

    if old not in dashboard:
        raise RuntimeError("Pi tab replacement did not match")
    dashboard = dashboard.replace(old, new, 1)

    bench_runner = r'''<div class="pi-bench-runner">
      <div class="section-title">Benchmark Runner</div>
      <div class="section-sub">Run model benchmark on a managed AI6 node</div>
      <div class="pi-bench-controls">
        <label>Node<select id="benchNode" onchange="benchNodeChanged()"></select></label>
        <label>Backend<select id="benchBackend" onchange="benchNodeChanged()"><option value="auto">Auto</option><option value="llama.cpp">llama.cpp</option><option value="ollama">Ollama</option><option value="vllm">vLLM</option></select></label>
        <label>Model<select id="benchModel"></select></label>
        <label>GPU<select id="benchGpu"></select></label>
        <label>Context<select id="benchCtx"><option>4096</option><option selected>8192</option><option>16384</option><option>32768</option></select></label>
        <label>Tokens<select id="benchTokens"><option>128</option><option selected>256</option><option>512</option></select></label>
        <button id="benchRunBtn" onclick="runModelBench()">▶ Run benchmark</button>
      </div>
      <div id="benchRunStatus" class="section-sub"></div>
      <div class="pi-bench-head"><b>Latest model runs</b><button onclick="loadPiBench()">Refresh</button></div>
      <div id="piBench" class="pi-bench"></div>
    </div>'''
    bench_markers = (
        '<div id="benchTab" style="display:none">',
        '<div id="benchsTab" style="display:none">',
        '<div id="benchTab">',
        '<div id="benchsTab">',
    )
    for bench_marker in bench_markers:
        if bench_marker in dashboard:
            dashboard = dashboard.replace(bench_marker, bench_marker + '<div class="section bench-runner-section">' + bench_runner + '</div>', 1)
            break
    else:
        raise RuntimeError("Benchs tab marker not found")
    dashboard = dashboard.replace(
        "</style></head><body>",
        r'''.pi-toolbar{display:flex;gap:7px;flex-wrap:wrap}.pi-bench-runner{width:100%;max-width:none;box-sizing:border-box;padding:2px;background:transparent}.bench-runner-section{width:100%;max-width:none;box-sizing:border-box;overflow:hidden}.pi-bench{width:100%;max-width:100%;overflow:auto}.pi-bench-controls{display:grid;grid-template-columns:minmax(110px,.7fr) minmax(120px,.8fr) minmax(260px,2fr) minmax(130px,.8fr) minmax(100px,.7fr) minmax(90px,.6fr) auto;gap:8px;align-items:end;margin-top:8px;width:100%;box-sizing:border-box}.pi-bench-controls label{font-size:10px;color:#aaa}.pi-bench-controls select{display:block;margin-top:3px;width:100%;min-width:0;max-width:none;padding:6px}.pi-bench-controls button{padding:7px 10px;white-space:nowrap}@media(max-width:1050px){.pi-bench-controls{grid-template-columns:repeat(3,minmax(0,1fr))}.pi-bench-controls button{width:100%}}.pi-bench-head{display:flex;justify-content:space-between;align-items:center;margin-top:10px}.pi-bench{overflow:auto;margin-top:6px}.pi-bench table{width:100%;border-collapse:collapse;font-size:11px}.pi-bench th,.pi-bench td{padding:5px 7px;border-bottom:1px solid #2d2d2d;text-align:left;white-space:nowrap}.pi-settings{margin-top:10px;padding:10px;border:1px solid #333;border-radius:10px;background:#141414}.pi-settings-row{display:grid;grid-template-columns:160px 1fr 180px 190px;gap:7px;margin-top:8px}.pi-settings-row input{padding:7px}.pi-node{padding:8px 0;border-bottom:1px solid #292929}.pi-node-start{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin-top:6px}.pi-node-start select{max-width:420px;padding:5px}.pi-gpu-policies{display:flex;gap:10px;flex-wrap:wrap;margin:5px 0;font-size:11px}.pi-gpu-policy{color:#bbb}@media(max-width:900px){.pi-settings-row{grid-template-columns:1fr}}.pi-toolbar select{min-width:190px}.pi-chat{height:590px;overflow:auto;background:#101010;border:1px solid #333;border-radius:12px;padding:14px;margin:12px 0}.pi-empty{color:#777;text-align:center;padding:70px 10px}.pi-msg{max-width:88%;margin:9px 0;padding:10px 12px;border-radius:11px;white-space:pre-wrap;line-height:1.45}.pi-user{margin-left:auto;background:#263044;border:1px solid #3b4b68}.pi-assistant{margin-right:auto;background:#181818;border:1px solid #333}.pi-meta{font-size:10px;color:#888;margin-top:7px}.pi-compose{display:grid;grid-template-columns:1fr 90px;gap:8px}.pi-compose textarea{resize:vertical;min-height:92px;padding:11px;background:#151515}.pi-compose button{font-weight:750}.pi-busy{color:#ffd166}@media(max-width:700px){.pi-chat{height:500px}.pi-msg{max-width:96%}.pi-compose{grid-template-columns:1fr}}
</style></head><body>''',
        1,
    )

    js = r'''
<script>
let piHistory=[];
function escPi(s){return String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function renderPi(){
 const box=document.getElementById('piChat');if(!box)return;
 if(!piHistory.length){box.innerHTML='<div class="pi-empty">Pi Coder is ready. Ask it to inspect, edit or test this repository.</div>';return}
 box.innerHTML=piHistory.map(m=>'<div class="pi-msg '+(m.role==='user'?'pi-user':'pi-assistant')+'">'+escPi(m.content)+(m.meta?'<div class="pi-meta">'+escPi(m.meta)+'</div>':'')+'</div>').join('');
 box.scrollTop=box.scrollHeight;
}
let piNodeCache=[];
async function loadPiBench(){
 try{
  const r=await fetch('/api/pi/benchmarks');const d=await r.json();const rows=d.runs||[];
  if(!rows.length){piBench.innerHTML='<span class="muted">No saved runs yet.</span>';return}
  const shown=rows.slice(0,7);
  piBench.innerHTML='<div style="max-height:255px;overflow:auto"><table><thead><tr><th>Node</th><th>Backend</th><th>Model</th><th>GPU</th><th>Prompt tok/s</th><th>Gen tok/s</th><th>VRAM</th><th>W</th><th>Temp</th><th>Total</th></tr></thead><tbody>'+shown.map(x=>{if(x.kind==='model-bench'){const gs=x.gpu||[];return '<tr><td>'+escPi(x.node)+'</td><td>'+escPi(x.backend)+'</td><td title="'+escPi(x.model)+'">'+escPi(String(x.model||'').split('/').pop())+'</td><td>'+escPi((x.gpus||[]).length?(x.gpus||[]).join('+'):'auto')+'</td><td>'+(x.prompt_tps??'—')+'</td><td>'+(x.gen_tps??(x.error?'ERR':'—'))+'</td><td>'+escPi(gs.map(g=>Math.round(g.memory_used_mib||0)+'M').join('+')||'—')+'</td><td>'+escPi(gs.map(g=>g.power_w??'—').join('+')||'—')+'</td><td>'+escPi(gs.map(g=>g.temp_c??'—').join('+')||'—')+'</td><td>'+(x.elapsed_s??'—')+'s</td></tr>'}return '<tr><td>Pi</td><td>agent</td><td>'+escPi(x.label)+'</td><td>—</td><td>—</td><td>'+(x.agent_tok_s??'—')+'</td><td>—</td><td>—</td><td>—</td><td>'+x.elapsed_s+'s</td>'}).join('')+'</tbody></table></div><div class="section-sub">Showing latest 7 of '+rows.length+' runs • scroll inside table</div>';
 }catch(e){piBench.textContent='Benchmark history error: '+e.message}
}
async function loadBenchNodes(){
 try{
  const d=await (await fetch('/api/pi/nodes?ts='+Date.now(),{cache:'no-store'})).json();
  piNodeCache=d.nodes||[];
  benchNode.innerHTML=piNodeCache.map(n=>'<option value="'+encodeURIComponent(n.name)+'">'+escPi(n.name)+'</option>').join('');
  await benchNodeChanged();
 }catch(e){benchRunStatus.textContent='Node list error: '+e.message}
}
async function benchNodeChanged(){
 const nodeSel=document.getElementById('benchNode'),modelSel0=document.getElementById('benchModel'),gpuSel0=document.getElementById('benchGpu');
 const nodeValue=nodeSel.value;
 modelSel0.innerHTML='<option value="">Loading models…</option>';gpuSel0.innerHTML='<option value="auto">Loading GPUs…</option>';
 try{
  const d=await (await fetch('/api/pi/nodes?ts='+Date.now(),{cache:'no-store'})).json();
  piNodeCache=d.nodes||[];
  const n=piNodeCache.find(x=>encodeURIComponent(x.name)===nodeValue);if(!n){benchModel.innerHTML='<option value="">No node data</option>';return}
  const backend=benchBackend.value,ms=[],seen=new Set();
  if(backend==='auto'||backend==='llama.cpp'){
   for(const m of [...(n.models||[]),...(n.running_models||[])]){
    if(!m||!m.path||seen.has('g:'+m.path))continue;seen.add('g:'+m.path);
    ms.push({v:'gguf:'+m.path,t:'GGUF • '+(m.label||String(m.path).split('/').pop())});
   }
  }
  if(backend==='auto'||backend==='ollama'){
   for(const m of (n.ollama_models||[])){if(!m||seen.has('o:'+m))continue;seen.add('o:'+m);ms.push({v:'ollama:'+m,t:'Ollama • '+m})}
  }
  // Do not HTML-escape option values/text while constructing DOM with innerHTML:
  // escPi is for chat rendering and was turning model paths into markup/empty options
  // in some browsers. Build real Option nodes instead.
  const modelSel=document.getElementById('benchModel'),gpuSel=document.getElementById('benchGpu');
  modelSel.options.length=0;
  if(ms.length){
   for(const m of ms){const o=document.createElement('option');o.value=m.v;o.text=m.t;modelSel.add(o)}
  }else{
   const o=document.createElement('option');o.value='';o.text='No models discovered';modelSel.add(o);
  }
  const dis=new Set((n.disabled_gpus||[]).map(Number));
  gpuSel.options.length=0;let go=document.createElement('option');go.value='auto';go.text='Auto';gpuSel.add(go);
  for(const g of (n.gpus||[])){go=document.createElement('option');go.value=String(g.index);go.text='GPU '+g.index+' • '+String(g.name||'');go.disabled=dis.has(Number(g.index));gpuSel.add(go)}
  if((n.gpus||[]).length>1){go=document.createElement('option');go.value='all';go.text='All enabled GPUs';gpuSel.add(go)}
  benchRunStatus.textContent='Discovered: '+(n.models||[]).length+' GGUF • '+(n.running_models||[]).length+' running • '+(n.ollama_models||[]).length+' Ollama';
 }catch(e){document.getElementById('benchModel').innerHTML='<option value="">Discovery error</option>';document.getElementById('benchGpu').innerHTML='<option value="auto">Auto</option>';document.getElementById('benchRunStatus').textContent='Discovery ERROR: '+e.message}
}
async function runModelBench(){
 const n=piNodeCache.find(x=>encodeURIComponent(x.name)===benchNode.value);if(!n||!benchModel.value)return;
 let gpus=[];if(benchGpu.value==='all'){const dis=new Set((n.disabled_gpus||[]).map(Number));gpus=(n.gpus||[]).map(g=>Number(g.index)).filter(x=>!dis.has(x))}else if(benchGpu.value!=='auto')gpus=[Number(benchGpu.value)];
 const payload={name:n.name,backend:benchBackend.value,model:benchModel.value,gpus,ctx:Number(benchCtx.value),tokens:Number(benchTokens.value),split:'layer'};
 benchRunBtn.disabled=true;benchRunStatus.textContent='Running '+n.name+' • '+benchModel.options[benchModel.selectedIndex].text+'…';
 try{const r=await fetch('/api/pi/benchmark/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));const x=d.run;benchRunStatus.textContent='Done • prompt '+(x.prompt_tps??'—')+' tok/s • gen '+(x.gen_tps??'—')+' tok/s • '+x.elapsed_s+'s';await loadPiBench();await loadPiNodes()}catch(e){benchRunStatus.textContent='ERROR: '+e.message;await loadPiBench()}finally{benchRunBtn.disabled=false}
}
async function loadPiNodes(){
 try{
  const r=await fetch('/api/pi/nodes');const d=await r.json();const ns=d.nodes||[];
  piNodes.innerHTML=ns.length?ns.map(n=>{
   const disabled=new Set((n.disabled_gpus||[]).map(Number));
   const gs=(n.gpus||[]).map(g=>'GPU'+g.index+' '+g.name+' '+Math.round(g.memory_total_mib||0)+'MiB').join(' • ');
   const policies=(n.gpus||[]).map(g=>'<label class="pi-gpu-policy"><input type="checkbox" '+(!disabled.has(Number(g.index))?'checked':'')+' onchange="setPiGpuPolicy(\''+encodeURIComponent(n.name)+'\','+g.index+',this.checked)"> GPU '+g.index+' Auto</label>').join(' ');
   const seen=new Set();const allModels=[...(n.models||[]),...(n.running_models||[])].filter(m=>m&&m.path&&!seen.has(m.path)&&seen.add(m.path));
   const ggufOpts=allModels.map(m=>'<option value="gguf:'+escPi(m.path)+'">GGUF • '+escPi(m.label||String(m.path).split('/').pop())+'</option>').join('');
   const ollamaOpts=(n.ollama_models||[]).map(m=>'<option value="ollama:'+escPi(m)+'">Ollama • '+escPi(m)+'</option>').join('');
   const opts=ggufOpts+ollamaOpts;
   const gpuChecks='<label><input type="radio" name="nodeGpu_'+encodeURIComponent(n.name)+'" class="nodeGpuAuto" data-node="'+encodeURIComponent(n.name)+'" value="auto" checked>Auto</label> '+(n.gpus||[]).map(g=>'<label><input type="radio" name="nodeGpu_'+encodeURIComponent(n.name)+'" class="nodeGpu" data-node="'+encodeURIComponent(n.name)+'" value="'+g.index+'" '+(disabled.has(Number(g.index))?'disabled':'')+'>GPU '+g.index+(disabled.has(Number(g.index))?' (disabled)':'')+'</label>').join(' ');
   const workers=Object.entries(n.workers||{}).map(([p,w])=>'<span>'+p+' '+escPi(w.status_text||w.state||'')+(w.managed?'':' <button onclick="stopPiNodeWorker(\''+encodeURIComponent(n.name)+'\','+p+')">Stop</button>')+'</span>').join(' • ');
   return '<div class="pi-node"><b>'+escPi(n.name)+'</b> • '+(n.online?'🟢':'🔴')+' '+escPi(n.monitor_url)+'<div class="section-sub">'+escPi(gs)+(n.error?' • '+escPi(n.error):'')+'</div><div class="pi-gpu-policies">'+policies+'</div>'+(n.online?'<div class="pi-node-start"><select id="nodeModel_'+encodeURIComponent(n.name)+'">'+opts+'</select><span>'+gpuChecks+'</span><select id="nodeCtx_'+encodeURIComponent(n.name)+'"><option>8192</option><option>16384</option><option>32768</option></select><button onclick="startPiNodeWorker(\''+encodeURIComponent(n.name)+'\')">Start + attach</button><button onclick="deletePiNode(\''+encodeURIComponent(n.name)+'\')">Delete node</button></div><div class="section-sub">'+workers+'</div>':'')+'</div>';
  }).join(''):'No managed nodes configured.';
 }catch(e){piNodes.textContent='Nodes error: '+e.message}
}
async function addPiNode(){
 const payload={name:piNodeName.value.trim(),monitor_url:piNodeUrl.value.trim()};if(!payload.name||!payload.monitor_url)return;
 piNodes.textContent='Testing node…';
 try{const r=await fetch('/api/pi/nodes',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));piNodeName.value='';piNodeUrl.value='';await loadPiNodes()}catch(e){piNodes.textContent='ERROR: '+e.message}
}
async function deletePiNode(name){await fetch('/api/pi/nodes/'+name,{method:'DELETE'});loadPiNodes()}
async function setPiGpuPolicy(enc,gpu,allow_auto){
 const name=decodeURIComponent(enc);
 try{const r=await fetch('/api/pi/nodes/gpu-policy',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,gpu,allow_auto})});const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));await loadPiNodes()}catch(e){alert('GPU policy: '+e.message);await loadPiNodes()}
}
async function startPiNodeWorker(enc){
 const name=decodeURIComponent(enc), rawModel=document.getElementById('nodeModel_'+enc).value,ctx=Number(document.getElementById('nodeCtx_'+enc).value);
 if(rawModel.startsWith('ollama:')){
  const model=rawModel.slice(7);const nodes=(await (await fetch('/api/pi/nodes')).json()).nodes||[];const n=nodes.find(x=>x.name===name);if(!n){alert('Node not found');return}
  const u=new URL(n.monitor_url);const base=u.protocol+'//'+u.hostname+':11434/v1';
  piStatus.textContent='● Attaching Ollama '+model+' @ '+name+'…';piStatus.className='status loading';
  try{const r=await fetch('/api/pi/providers',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:name+'-ollama',base_url:base,api_key:'ollama'})});const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));await loadPiWorkers();const key='ai6-web-'+(name+'-ollama').toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'');const wanted=key+'::'+model;if([...piModel.options].some(x=>x.value===wanted))piModel.value=wanted;piStatus.textContent='● Ollama ready • '+model+' @ '+name;piStatus.className='status ready'}catch(e){piStatus.textContent='● Ollama attach error: '+e.message;piStatus.className='status error'}return
 }
 const model=rawModel.startsWith('gguf:')?rawModel.slice(5):rawModel;
 const gpus=[...document.querySelectorAll('.nodeGpu[data-node="'+enc+'"]:checked')].map(x=>Number(x.value));
 piStatus.textContent='● Starting '+name+' worker…';piStatus.className='status loading';
 try{const r=await fetch('/api/pi/nodes/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,model,gpus,split:'layer',ctx,ngl:999,alias:'pi'})});const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));await loadPiNodes();await loadPiWorkers();if(d.ready&&d.provider){const key='ai6-web-'+d.provider.toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'');const wanted=key+'::'+(d.models||[])[0];if([...piModel.options].some(x=>x.value===wanted))piModel.value=wanted}piStatus.textContent=d.ready?'● Remote worker ready • '+d.provider+' • GPU '+(d.gpus||[]).join(','):'● Worker started on '+d.port+' • still loading';piStatus.className=d.ready?'status ready':'status loading'}catch(e){piStatus.textContent='● Remote start error: '+e.message;piStatus.className='status error'}
}
async function stopPiNodeWorker(enc,port){
 const name=decodeURIComponent(enc);const r=await fetch('/api/pi/nodes/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,port})});const d=await r.json();if(!r.ok){alert(d.detail||JSON.stringify(d));return}await loadPiNodes();await loadPiWorkers()
}

async function loadPiWorkers(){
 try{
  const r=await fetch('/api/pi/providers');const d=await r.json();const ps=d.providers||[];
  piWorkers.innerHTML=ps.length?ps.map(p=>'<div><b>'+escPi(p.name)+'</b> • '+escPi(p.base_url)+' • '+p.models.length+' model(s) <button data-del="'+encodeURIComponent(p.name)+'">Delete</button></div>').join(''):'No remote workers configured.';
  piWorkers.querySelectorAll('button[data-del]').forEach(b=>b.onclick=()=>deletePiWorker(decodeURIComponent(b.dataset.del)));
  const sel=document.getElementById('piModel'),current=sel.value;sel.querySelectorAll('option[data-remote]').forEach(x=>x.remove());
  ps.forEach(p=>{const key='ai6-web-'+p.name.toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'');p.models.forEach(mid=>{const o=document.createElement('option');o.value=key+'::'+mid;o.textContent=mid+' @ '+p.name;o.dataset.remote='1';sel.appendChild(o)})});if([...sel.options].some(x=>x.value===current))sel.value=current;
 }catch(e){piWorkers.textContent='Workers error: '+e.message}
}
function togglePiSettings(){const x=document.getElementById('piSettings');x.style.display=x.style.display==='none'?'block':'none';if(x.style.display==='block')loadPiWorkers()}
async function addPiWorker(){
 const payload={name:piWorkerName.value.trim(),base_url:piWorkerUrl.value.trim(),api_key:piWorkerKey.value.trim()||'ollama'};if(!payload.name||!payload.base_url)return;
 piWorkers.textContent='Testing '+payload.base_url+'…';
 try{const r=await fetch('/api/pi/providers',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));await loadPiWorkers();piWorkerName.value='';piWorkerUrl.value='';piWorkerKey.value=''}catch(e){piWorkers.textContent='ERROR: '+e.message}
}
async function deletePiWorker(name){await fetch('/api/pi/providers/'+encodeURIComponent(name),{method:'DELETE'});loadPiWorkers()}
async function refreshPiStatus(){
 try{const r=await fetch('/api/pi/status');const s=await r.json();const installed=(s.models||[]).filter(x=>x.installed).map(x=>x.label).join(', ');piStatus.textContent=(s.pi_installed?'● Pi ready':'● Pi missing')+' • '+s.repo+' • models: '+(installed||'none');piStatus.className=s.pi_installed?'status ready':'status down'}catch(e){piStatus.textContent='Pi status error: '+e.message}
}
async function sendPi(){
 const input=document.getElementById('piPrompt'),btn=document.getElementById('piSend'),model=document.getElementById('piModel').value;const prompt=input.value.trim();if(!prompt)return;
 const prior=piHistory.map(x=>({role:x.role,content:x.content}));
 piHistory.push({role:'user',content:prompt});renderPi();input.value='';btn.disabled=true;piStatus.textContent='● Pi working…';piStatus.className='status loading';
 try{
  const r=await fetch('/api/pi/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt,model,history:prior})});const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  const u=d.usage||{};const speed=d.avg_output_tok_s!=null?d.avg_output_tok_s+' agent tok/s':'tok/s —';const toks=(u.input||0)+' in / '+(u.output||0)+' out';const tools=(d.tool_calls||0)+' tools';piHistory.push({role:'assistant',content:d.response||'(no text)',meta:d.label+' • '+d.elapsed_s+' s • tools '+(d.tool_time_s??0)+' s • LLM+overhead '+(d.llm_plus_overhead_s??d.elapsed_s)+' s • '+toks+' • '+speed+' • '+tools});renderPi();loadPiBench();piStatus.textContent='● Pi ready • '+d.label+' • '+toks+' • '+speed;piStatus.className='status ready';
 }catch(e){piHistory.push({role:'assistant',content:'ERROR: '+e.message,meta:'request failed'});renderPi();piStatus.textContent='● Pi error';piStatus.className='status error'}
 finally{btn.disabled=false;input.focus()}
}
function clearPiChat(){piHistory=[];renderPi();document.getElementById('piPrompt').focus()}
function openPiTerminal(){window.open(location.protocol+'//'+location.hostname+':8093/','_blank','noopener')}
window.addEventListener('load',()=>{const p=document.getElementById('piPrompt');if(p)p.addEventListener('keydown',e=>{if(e.key==='Enter'&&e.ctrlKey){e.preventDefault();sendPi()}});refreshPiStatus();loadPiWorkers();loadPiNodes();loadBenchNodes();loadPiBench()});
</script>
'''
    dashboard = dashboard.replace("</body>", js + "\n</body>", 1)
    base.DASHBOARD = dashboard

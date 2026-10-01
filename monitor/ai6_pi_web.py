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
            row.update({"online": True, "host": stats.get("host", {}),
                        "gpus": stats.get("gpu", {}).get("devices", []),
                        "workers": stats.get("llama", {}).get("workers", {}),
                        "models": models, "ollama_models": ollama_models})
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
    <div class="pi-bench-head"><b>Recent agent runs</b><button onclick="loadPiBench()">Refresh</button></div>
    <div id="piBench" class="pi-bench"></div>
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
    dashboard = dashboard.replace(
        "</style></head><body>",
        r'''.pi-toolbar{display:flex;gap:7px;flex-wrap:wrap}.pi-bench-head{display:flex;justify-content:space-between;align-items:center;margin-top:10px}.pi-bench{overflow:auto;margin-top:6px}.pi-bench table{width:100%;border-collapse:collapse;font-size:11px}.pi-bench th,.pi-bench td{padding:5px 7px;border-bottom:1px solid #2d2d2d;text-align:left;white-space:nowrap}.pi-settings{margin-top:10px;padding:10px;border:1px solid #333;border-radius:10px;background:#141414}.pi-settings-row{display:grid;grid-template-columns:160px 1fr 180px 190px;gap:7px;margin-top:8px}.pi-settings-row input{padding:7px}.pi-node{padding:8px 0;border-bottom:1px solid #292929}.pi-node-start{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin-top:6px}.pi-node-start select{max-width:420px;padding:5px}.pi-gpu-policies{display:flex;gap:10px;flex-wrap:wrap;margin:5px 0;font-size:11px}.pi-gpu-policy{color:#bbb}@media(max-width:900px){.pi-settings-row{grid-template-columns:1fr}}.pi-toolbar select{min-width:190px}.pi-chat{height:590px;overflow:auto;background:#101010;border:1px solid #333;border-radius:12px;padding:14px;margin:12px 0}.pi-empty{color:#777;text-align:center;padding:70px 10px}.pi-msg{max-width:88%;margin:9px 0;padding:10px 12px;border-radius:11px;white-space:pre-wrap;line-height:1.45}.pi-user{margin-left:auto;background:#263044;border:1px solid #3b4b68}.pi-assistant{margin-right:auto;background:#181818;border:1px solid #333}.pi-meta{font-size:10px;color:#888;margin-top:7px}.pi-compose{display:grid;grid-template-columns:1fr 90px;gap:8px}.pi-compose textarea{resize:vertical;min-height:92px;padding:11px;background:#151515}.pi-compose button{font-weight:750}.pi-busy{color:#ffd166}@media(max-width:700px){.pi-chat{height:500px}.pi-msg{max-width:96%}.pi-compose{grid-template-columns:1fr}}
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
async function loadPiBench(){
 try{
  const r=await fetch('/api/pi/benchmarks');const d=await r.json();const rows=d.runs||[];
  if(!rows.length){piBench.innerHTML='<span class="muted">No saved runs yet.</span>';return}
  piBench.innerHTML='<table><thead><tr><th>Model / worker</th><th>Total</th><th>Tool</th><th>LLM+overhead</th><th>In</th><th>Out</th><th>Agent tok/s</th><th>Tools</th></tr></thead><tbody>'+rows.slice(0,12).map(x=>'<tr><td>'+escPi(x.label)+'</td><td>'+x.elapsed_s+'s</td><td>'+x.tool_time_s+'s</td><td>'+x.llm_plus_overhead_s+'s</td><td>'+x.input+'</td><td>'+x.output+'</td><td>'+(x.agent_tok_s??'—')+'</td><td>'+x.tool_calls+'</td></tr>').join('')+'</tbody></table>';
 }catch(e){piBench.textContent='Benchmark history error: '+e.message}
}
async function loadPiNodes(){
 try{
  const r=await fetch('/api/pi/nodes');const d=await r.json();const ns=d.nodes||[];
  piNodes.innerHTML=ns.length?ns.map(n=>{
   const disabled=new Set((n.disabled_gpus||[]).map(Number));
   const gs=(n.gpus||[]).map(g=>'GPU'+g.index+' '+g.name+' '+Math.round(g.memory_total_mib||0)+'MiB').join(' • ');
   const policies=(n.gpus||[]).map(g=>'<label class="pi-gpu-policy"><input type="checkbox" '+(!disabled.has(Number(g.index))?'checked':'')+' onchange="setPiGpuPolicy(\''+encodeURIComponent(n.name)+'\','+g.index+',this.checked)"> GPU '+g.index+' Auto</label>').join(' ');
   const ggufOpts=(n.models||[]).map(m=>'<option value="gguf:'+escPi(m.path)+'">GGUF • '+escPi(m.label)+'</option>').join('');
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
window.addEventListener('load',()=>{const p=document.getElementById('piPrompt');if(p)p.addEventListener('keydown',e=>{if(e.key==='Enter'&&e.ctrlKey){e.preventDefault();sendPi()}});refreshPiStatus();loadPiWorkers();loadPiNodes();loadPiBench()});
</script>
'''
    dashboard = dashboard.replace("</body>", js + "\n</body>", 1)
    base.DASHBOARD = dashboard

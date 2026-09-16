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
from pydantic import BaseModel

import ai6_monitor as base


_ORIGINAL_COLLECT_STATS = base.collect_stats
_LOG_TAIL_LINES = 160
_SUBPROCESS_TIMEOUT = 1.5
_MODELS_TIMEOUT = 0.7
_MODELS_DIR = Path(os.environ.get("AI6_MODELS_DIR", str(Path.home() / "models")))

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


class PresetCommand(BaseModel):
    profile: str
    model: str
    split: str = "layer"
    ctx: int = 32768


@base.app.get("/api/models")
def api_models():
    models = []
    if _MODELS_DIR.exists():
        for path in sorted(_MODELS_DIR.rglob("*.gguf")):
            try:
                rel = path.relative_to(_MODELS_DIR)
            except ValueError:
                rel = path.name
            label = str(rel)
            models.append({"label": label, "path": str(path)})
    return {"models": models}


@base.app.post("/api/workers/preset")
def api_worker_preset(cmd: PresetCommand):
    if cmd.profile not in ("33", "222"):
        raise base.HTTPException(status_code=400, detail="invalid profile")
    if cmd.split not in ("tensor", "layer"):
        raise base.HTTPException(status_code=400, detail="invalid split")
    if not (2048 <= cmd.ctx <= 262144):
        raise base.HTTPException(status_code=400, detail="invalid context")
    model = Path(cmd.model)
    try:
        model.resolve().relative_to(_MODELS_DIR.resolve())
    except Exception:
        raise base.HTTPException(status_code=400, detail="model must be below models directory")
    if not model.is_file() or model.suffix.lower() != ".gguf":
        raise base.HTTPException(status_code=400, detail="model file not found")
    try:
        out = base.run_workerctl("preset", cmd.profile, str(model), cmd.split, str(cmd.ctx))
        return {"ok": True, "output": out}
    except RuntimeError as e:
        raise base.HTTPException(status_code=409, detail=str(e))


_dashboard = base.DASHBOARD
_dashboard = _dashboard.replace(
    "Model, GPU assignment, readiness and last measured throughput",
    "Auto-discovered llama.cpp processes: model, GPUs, context, readiness and throughput",
)

_dashboard = _dashboard.replace(
    '<div class="profile-actions"><button onclick="setProfile(\'33\')">3+3</button><button onclick="setProfile(\'222\')">2+2+2</button></div>',
    '''<div class="preset-wrap">
      <div class="preset-card-ui">
        <button class="preset-main" onclick="applyPreset('222')"><b>2+2+2 — Daily Coding</b><small>3 workers · GPU 0+1 / 2+3 / 4+5</small></button>
        <select id="preset222model" class="preset-model" title="Model for 2+2+2"></select>
        <div class="preset-foot">layer · 32K</div>
      </div>
      <div class="preset-card-ui">
        <button class="preset-main" onclick="applyPreset('33')"><b>3+3 — Large Context</b><small>2 workers · GPU 0+1+2 / 3+4+5</small></button>
        <select id="preset33model" class="preset-model" title="Model for 3+3"></select>
        <div class="preset-foot">layer · 32K</div>
      </div>
    </div>''',
)

_dashboard = _dashboard.replace(
    "</style></head><body>",
    '''.preset-wrap{display:flex;gap:8px;flex-wrap:wrap}.preset-card-ui{min-width:210px;background:#161616;border:1px solid #333;border-radius:10px;padding:7px}.preset-main{width:100%;text-align:left;padding:7px 9px}.preset-main b{display:block;font-size:12px}.preset-main small{display:block;color:#aaa;font-size:10px;margin-top:2px}.preset-model{width:100%;margin-top:5px;padding:5px 7px;font-size:10px;background:#202020}.preset-foot{font-size:10px;color:#888;margin:4px 2px 0}.preset-card-ui.busy{opacity:.6;pointer-events:none}
</style></head><body>''',
)

_preset_js = r'''
let presetModelsLoaded=false;
async function loadPresetModels(){
 if(presetModelsLoaded)return;
 try{
  const r=await fetch('/api/models');const d=await r.json();
  const models=d.models||[];
  for(const id of ['preset222model','preset33model']){
   const el=document.getElementById(id);if(!el)continue;
   el.innerHTML=models.map(m=>'<option value="'+m.path.replace(/"/g,'&quot;')+'">'+m.label+'</option>').join('');
   const preferred=models.findIndex(m=>m.label.toLowerCase().includes('qwen2.5-coder-14b'));
   if(preferred>=0)el.selectedIndex=preferred;
  }
  presetModelsLoaded=true;
 }catch(e){workermsg.textContent='Model list error: '+e;}
}
async function applyPreset(profileId){
 const select=document.getElementById(profileId==='222'?'preset222model':'preset33model');
 if(!select||!select.value){workermsg.textContent='Select a GGUF model first';return;}
 if(!confirm('Switch workers to '+(profileId==='222'?'2+2+2 Daily Coding':'3+3 Large Context')+' using '+select.options[select.selectedIndex].text+'?'))return;
 workermsg.textContent='Applying preset…';
 try{
  const r=await fetch('/api/workers/preset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({profile:profileId,model:select.value,split:'layer',ctx:32768})});
  const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  workermsg.textContent='Preset applied: '+(d.output||'OK');setTimeout(refresh,800);
 }catch(e){workermsg.textContent='Preset error: '+e.message;}
}
'''

_worker_js = _preset_js + r'''function dynamicWorkerCard(p,w){
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
 loadPresetModels();
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

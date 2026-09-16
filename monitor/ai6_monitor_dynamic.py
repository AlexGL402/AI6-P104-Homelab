#!/usr/bin/env python3
"""Dynamic worker discovery and flexible worker launcher for the AI6 monitor."""

import json
import os
import re
import signal
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
_LLAMA_BIN = Path(os.environ.get("AI6_LLAMA_BIN", str(Path.home() / "llama.cpp/build/bin/llama-server")))
_CUSTOM_LOG_DIR = Path(os.environ.get("AI6_CUSTOM_LOG_DIR", str(Path.home() / ".local/state/ai6-monitor/workers")))

_CLI_OPTIONS = {
    "--port": "port", "-p": "port",
    "--ctx-size": "ctx", "-c": "ctx",
    "--alias": "alias",
    "--model": "model", "-m": "model",
    "--split-mode": "split", "-sm": "split",
    "--gpu-layers": "ngl", "-ngl": "ngl",
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
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time", "username"]):
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
    ngl = _parse_int(options.get("ngl"))
    split = options.get("split")
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
        "ngl": ngl,
        "split": split,
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
    return {str(port): _dynamic_worker_details(port, proc) for port, proc in _worker_processes().items()}


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


class PresetControlCommand(BaseModel):
    profile: str
    action: str


class CustomWorkerCommand(BaseModel):
    model: str
    gpus: list[int]
    split: str = "layer"
    ctx: int = 32768
    ngl: int = 999
    port: int | None = None
    alias: str = "custom"


class CustomStopCommand(BaseModel):
    port: int


def _validate_model(model_text):
    model = Path(model_text)
    try:
        model.resolve().relative_to(_MODELS_DIR.resolve())
    except Exception:
        raise base.HTTPException(status_code=400, detail="model must be below models directory")
    if not model.is_file() or model.suffix.lower() != ".gguf":
        raise base.HTTPException(status_code=400, detail="model file not found")
    return model.resolve()


def _pick_custom_port(requested=None):
    used = set(_worker_processes())
    if requested is not None:
        if not (8100 <= requested <= 8199):
            raise base.HTTPException(status_code=400, detail="custom port must be 8100..8199")
        if requested in used or base.service_ok(requested):
            raise base.HTTPException(status_code=409, detail=f"port {requested} is already in use")
        return requested
    for port in range(8101, 8200):
        if port not in used and not base.service_ok(port):
            return port
    raise base.HTTPException(status_code=409, detail="no free custom port in 8101..8199")


@base.app.get("/api/models")
def api_models():
    models = []
    if _MODELS_DIR.exists():
        for path in sorted(_MODELS_DIR.rglob("*.gguf")):
            try:
                rel = path.relative_to(_MODELS_DIR)
            except ValueError:
                rel = path.name
            models.append({"label": str(rel), "path": str(path)})
    return {"models": models}


@base.app.post("/api/workers/preset")
def api_worker_preset(cmd: PresetCommand):
    if cmd.profile not in ("33", "222"):
        raise base.HTTPException(status_code=400, detail="invalid profile")
    if cmd.split not in ("tensor", "layer"):
        raise base.HTTPException(status_code=400, detail="invalid split")
    if not (2048 <= cmd.ctx <= 262144):
        raise base.HTTPException(status_code=400, detail="invalid context")
    model = _validate_model(cmd.model)
    try:
        out = base.run_workerctl("preset", cmd.profile, str(model), cmd.split, str(cmd.ctx))
        return {"ok": True, "output": out}
    except RuntimeError as e:
        raise base.HTTPException(status_code=409, detail=str(e))


@base.app.post("/api/workers/preset-control")
def api_worker_preset_control(cmd: PresetControlCommand):
    if cmd.profile not in ("33", "222") or cmd.action not in ("start", "stop", "restart"):
        raise base.HTTPException(status_code=400, detail="invalid preset control")
    try:
        out = base.run_workerctl("preset-control", cmd.profile, cmd.action)
        return {"ok": True, "output": out}
    except RuntimeError as e:
        raise base.HTTPException(status_code=409, detail=str(e))


@base.app.post("/api/workers/custom/start")
def api_custom_start(cmd: CustomWorkerCommand):
    model = _validate_model(cmd.model)
    if cmd.split not in ("tensor", "layer"):
        raise base.HTTPException(status_code=400, detail="split must be tensor or layer")
    if not (2048 <= cmd.ctx <= 262144):
        raise base.HTTPException(status_code=400, detail="invalid context")
    if not (0 <= cmd.ngl <= 999):
        raise base.HTTPException(status_code=400, detail="invalid ngl")
    if not cmd.gpus or len(cmd.gpus) > 6 or len(set(cmd.gpus)) != len(cmd.gpus):
        raise base.HTTPException(status_code=400, detail="select 1..6 unique GPUs")
    gpu_count = len(base.gpu_stats())
    if any(g < 0 or g >= gpu_count for g in cmd.gpus):
        raise base.HTTPException(status_code=400, detail="GPU index out of range")
    alias = re.sub(r"[^A-Za-z0-9_.-]+", "-", cmd.alias or "custom").strip("-")[:48] or "custom"
    port = _pick_custom_port(cmd.port)
    if not _LLAMA_BIN.is_file():
        raise base.HTTPException(status_code=409, detail=f"llama-server not found: {_LLAMA_BIN}")

    args = [
        str(_LLAMA_BIN), "-m", str(model), "-ngl", str(cmd.ngl),
        "-sm", cmd.split, "-c", str(cmd.ctx), "-np", "1",
        "--alias", alias, "--host", "0.0.0.0", "--port", str(port),
    ]
    if cmd.split == "tensor" and len(cmd.gpus) > 1:
        args += ["-ts", ",".join("1" for _ in cmd.gpus)]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in cmd.gpus)
    _CUSTOM_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _CUSTOM_LOG_DIR / f"worker-{port}.log"
    try:
        log = log_path.open("ab", buffering=0)
        proc = subprocess.Popen(
            args,
            cwd=str(_LLAMA_BIN.parent.parent.parent),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        log.close()
    except Exception as e:
        raise base.HTTPException(status_code=409, detail=f"failed to start worker: {e}")
    return {
        "ok": True, "pid": proc.pid, "port": port, "log": str(log_path),
        "gpus": cmd.gpus, "model": str(model), "split": cmd.split,
        "ctx": cmd.ctx, "ngl": cmd.ngl,
    }


@base.app.post("/api/workers/custom/stop")
def api_custom_stop(cmd: CustomStopCommand):
    workers = _worker_processes()
    proc = workers.get(cmd.port)
    if proc is None:
        raise base.HTTPException(status_code=404, detail="worker not found")
    if base._service_for_port(cmd.port):
        raise base.HTTPException(status_code=409, detail="managed preset worker; use preset/worker controls")
    try:
        username = proc.username()
        current = psutil.Process().username()
        if username != current:
            raise base.HTTPException(status_code=403, detail="worker belongs to another user")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        try:
            proc.wait(timeout=6)
        except psutil.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
    except base.HTTPException:
        raise
    except Exception as e:
        raise base.HTTPException(status_code=409, detail=f"failed to stop worker: {e}")
    return {"ok": True, "port": cmd.port}


_dashboard = base.DASHBOARD
_dashboard = _dashboard.replace(
    "Model, GPU assignment, readiness and last measured throughput",
    "Auto-discovered llama.cpp processes: model, GPUs, context, readiness and throughput",
)

_dashboard = _dashboard.replace(
    '<div class="profile-actions"><button onclick="setProfile(\'33\')">3+3</button><button onclick="setProfile(\'222\')">2+2+2</button></div>',
    '''<div class="preset-wrap">
      <div class="preset-card-ui">
        <div class="preset-title"><b>2+2+2 — Daily Coding</b><small>3 workers · GPU 0+1 / 2+3 / 4+5</small></div>
        <select id="preset222model" class="preset-model" title="Model for 2+2+2"></select>
        <div class="preset-foot">layer · 32K</div>
        <div class="preset-actions"><button onclick="applyPreset('222')">Start / Apply</button><button onclick="presetControl('222','stop')">Stop</button><button onclick="applyPreset('222')">Restart</button></div>
      </div>
      <div class="preset-card-ui">
        <div class="preset-title"><b>3+3 — Large Context</b><small>2 workers · GPU 0+1+2 / 3+4+5</small></div>
        <select id="preset33model" class="preset-model" title="Model for 3+3"></select>
        <div class="preset-foot">layer · 32K</div>
        <div class="preset-actions"><button onclick="applyPreset('33')">Start / Apply</button><button onclick="presetControl('33','stop')">Stop</button><button onclick="applyPreset('33')">Restart</button></div>
      </div>
    </div>''',
)

_custom_html = '''<div class="custom-builder">
  <div class="custom-title"><b>Custom Worker</b><span>Any GPU combination: 1, 1+1, 2, 3, 4…</span></div>
  <div class="custom-row">
    <label>Model<select id="customModel" class="custom-model"></select></label>
    <label>Split<select id="customSplit"><option value="layer">layer</option><option value="tensor">tensor</option></select></label>
    <label>Context<select id="customCtx"><option>8192</option><option>16384</option><option selected>32768</option><option>65536</option><option>98304</option></select></label>
    <label>NGL<input id="customNgl" type="number" min="0" max="999" value="999" title="GPU layers; e.g. 38 for 14B on one P104"></label>
    <label>Port<input id="customPort" type="number" min="8100" max="8199" placeholder="auto"></label>
    <label>Alias<input id="customAlias" value="custom"></label>
  </div>
  <div class="custom-gpus"><span>GPUs:</span>
    <label><input type="checkbox" name="customGpu" value="0">0</label>
    <label><input type="checkbox" name="customGpu" value="1">1</label>
    <label><input type="checkbox" name="customGpu" value="2">2</label>
    <label><input type="checkbox" name="customGpu" value="3">3</label>
    <label><input type="checkbox" name="customGpu" value="4">4</label>
    <label><input type="checkbox" name="customGpu" value="5">5</label>
    <button onclick="startCustomWorker()">Start custom worker</button>
    <span class="custom-hint">For two separate 1-GPU workers: start once on GPU 0, then again on GPU 1. Port can stay Auto.</span>
  </div>
</div>'''
_dashboard = _dashboard.replace('<div id="workerbuttons" class="worker-grid"></div>', _custom_html + '<div id="workerbuttons" class="worker-grid"></div>')

_dashboard = _dashboard.replace(
    "</style></head><body>",
    '''.preset-wrap{display:flex;gap:8px;flex-wrap:wrap}.preset-card-ui{min-width:270px;background:#161616;border:1px solid #333;border-radius:10px;padding:8px}.preset-title{padding:2px 3px}.preset-title b{display:block;font-size:12px}.preset-title small{display:block;color:#aaa;font-size:10px;margin-top:2px}.preset-model{width:100%;margin-top:5px;padding:5px 7px;font-size:10px;background:#202020}.preset-foot{font-size:10px;color:#888;margin:4px 2px 0}.preset-actions{display:flex;gap:5px;margin-top:7px}.preset-actions button{padding:5px 8px;font-size:10px}.custom-builder{margin:0 0 12px;padding:10px;background:#151515;border:1px solid #343434;border-radius:10px}.custom-title{display:flex;gap:8px;align-items:baseline;margin-bottom:8px}.custom-title span,.custom-hint{font-size:10px;color:#999}.custom-row{display:grid;grid-template-columns:minmax(260px,2fr) repeat(5,minmax(85px,1fr));gap:7px}.custom-row label{font-size:10px;color:#aaa}.custom-row select,.custom-row input{display:block;width:100%;margin-top:3px;padding:6px;font-size:11px}.custom-gpus{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:9px;font-size:11px}.custom-gpus label{background:#202020;border:1px solid #333;border-radius:7px;padding:5px 8px}.custom-gpus input{margin-right:4px}.custom-gpus button{padding:6px 10px;font-size:11px}@media(max-width:1000px){.custom-row{grid-template-columns:1fr 1fr 1fr}}
</style></head><body>''',
)

_extra_js = r'''
let presetModelsLoaded=false;
let modelCache=[];
async function loadPresetModels(){
 if(presetModelsLoaded)return;
 try{
  const r=await fetch('/api/models');const d=await r.json();modelCache=d.models||[];
  for(const id of ['preset222model','preset33model','customModel']){
   const el=document.getElementById(id);if(!el)continue;
   el.innerHTML=modelCache.map(m=>'<option value="'+m.path.replace(/"/g,'&quot;')+'">'+m.label+'</option>').join('');
   const preferred=modelCache.findIndex(m=>m.label.toLowerCase().includes('qwen2.5-coder-14b'));
   if(preferred>=0)el.selectedIndex=preferred;
  }
  presetModelsLoaded=true;
 }catch(e){workermsg.textContent='Model list error: '+e;}
}
async function applyPreset(profileId){
 const select=document.getElementById(profileId==='222'?'preset222model':'preset33model');
 if(!select||!select.value){workermsg.textContent='Select a GGUF model first';return;}
 const title=profileId==='222'?'2+2+2 Daily Coding':'3+3 Large Context';
 if(!confirm('Start '+title+' using '+select.options[select.selectedIndex].text+'?'))return;
 workermsg.textContent='Starting '+title+'…';
 try{
  const r=await fetch('/api/workers/preset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({profile:profileId,model:select.value,split:'layer',ctx:32768})});
  const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  workermsg.textContent='Preset started: '+(d.output||'OK');setTimeout(refresh,900);
 }catch(e){workermsg.textContent='Preset error: '+e.message;}
}
async function presetControl(profileId,action){
 if(!confirm(action+' preset '+(profileId==='222'?'2+2+2':'3+3')+'?'))return;
 workermsg.textContent=action+' preset…';
 try{
  const r=await fetch('/api/workers/preset-control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({profile:profileId,action:action})});
  const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  workermsg.textContent=d.output||'OK';setTimeout(refresh,700);
 }catch(e){workermsg.textContent='Preset control error: '+e.message;}
}
async function startCustomWorker(){
 const model=document.getElementById('customModel').value;
 const gpus=[...document.querySelectorAll('input[name="customGpu"]:checked')].map(x=>Number(x.value));
 if(!model){workermsg.textContent='Select a model';return;}
 if(!gpus.length){workermsg.textContent='Select at least one GPU';return;}
 const portText=document.getElementById('customPort').value.trim();
 const payload={model:model,gpus:gpus,split:document.getElementById('customSplit').value,ctx:Number(document.getElementById('customCtx').value),ngl:Number(document.getElementById('customNgl').value),port:portText?Number(portText):null,alias:document.getElementById('customAlias').value||'custom'};
 workermsg.textContent='Starting custom worker on GPU '+gpus.join('+')+'…';
 try{
  const r=await fetch('/api/workers/custom/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  workermsg.textContent='Custom worker started: port '+d.port+' • PID '+d.pid+' • GPU '+gpus.join('+');document.getElementById('customPort').value='';setTimeout(refresh,900);
 }catch(e){workermsg.textContent='Custom worker error: '+e.message;}
}
async function stopCustomWorker(port){
 if(!confirm('Stop worker '+port+'?'))return;
 try{
  const r=await fetch('/api/workers/custom/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({port:Number(port)})});const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  workermsg.textContent='Worker '+port+' stopped';setTimeout(refresh,500);
 }catch(e){workermsg.textContent='Stop error: '+e.message;}
}
'''

_worker_js = _extra_js + r'''function dynamicWorkerCard(p,w){
 const model=w.model||'llama-server';
 const gen=w.last_tok_s!=null?w.last_tok_s.toFixed(2):'—';
 const prompt=w.last_prompt_tok_s!=null?w.last_prompt_tok_s.toFixed(1):'—';
 const promptSize=w.last_prompt_tokens!=null?w.last_prompt_tokens:'—';
 const up=w.uptime_s!=null?fmtUptime(w.uptime_s):'—';
 const gpu=w.gpu_label||'GPU auto/all';
 const ctx=w.ctx!=null?w.ctx.toLocaleString():'—';
 const split=w.split||'—';const ngl=w.ngl!=null?w.ngl:'—';
 const meta=gpu+' • '+split+' • ctx '+ctx+' • ngl '+ngl+' • PID '+(w.pid??'—');
 const actions=w.managed
   ? '<div class="worker-actions"><button onclick="workerAction(\\'start\\','+p+')">Start</button><button onclick="workerAction(\\'stop\\','+p+')">Stop</button><button onclick="workerAction(\\'restart\\','+p+')">Restart</button></div>'
   : '<div class="worker-actions"><button onclick="stopCustomWorker('+p+')">Stop process</button></div>';
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

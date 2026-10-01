#!/usr/bin/env python3
"""Web UI/API for the local Pi coding agent."""

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

from pydantic import BaseModel, Field
import ai6_monitor_dynamic as dynamic

base = dynamic.base

REPO = Path.home() / "AI6-P104-Homelab"
PI = Path("/usr/bin/pi")
PROVIDERS_FILE = Path.home() / ".local/state/ai6-monitor/pi-providers.json"
BENCH_FILE = Path.home() / ".local/state/ai6-monitor/pi-benchmarks.jsonl"
MODELS = {
    "8b": ("ai6-ollama", "qwen3:8b", "Qwen3 8B"),
    "14b": ("ai6-ollama", "qwen3:14b", "Qwen3 14B"),
}


class ProviderCommand(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    base_url: str = Field(min_length=8, max_length=256)
    api_key: str = Field(default="ollama", max_length=256)


class PiChatCommand(BaseModel):
    prompt: str = Field(min_length=1, max_length=12000)
    model: str = Field(default="8b", max_length=256)
    history: list[dict] = Field(default_factory=list)


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
        r'''.pi-toolbar{display:flex;gap:7px;flex-wrap:wrap}.pi-bench-head{display:flex;justify-content:space-between;align-items:center;margin-top:10px}.pi-bench{overflow:auto;margin-top:6px}.pi-bench table{width:100%;border-collapse:collapse;font-size:11px}.pi-bench th,.pi-bench td{padding:5px 7px;border-bottom:1px solid #2d2d2d;text-align:left;white-space:nowrap}.pi-settings{margin-top:10px;padding:10px;border:1px solid #333;border-radius:10px;background:#141414}.pi-settings-row{display:grid;grid-template-columns:160px 1fr 180px 190px;gap:7px;margin-top:8px}.pi-settings-row input{padding:7px}@media(max-width:900px){.pi-settings-row{grid-template-columns:1fr}}.pi-toolbar select{min-width:190px}.pi-chat{height:590px;overflow:auto;background:#101010;border:1px solid #333;border-radius:12px;padding:14px;margin:12px 0}.pi-empty{color:#777;text-align:center;padding:70px 10px}.pi-msg{max-width:88%;margin:9px 0;padding:10px 12px;border-radius:11px;white-space:pre-wrap;line-height:1.45}.pi-user{margin-left:auto;background:#263044;border:1px solid #3b4b68}.pi-assistant{margin-right:auto;background:#181818;border:1px solid #333}.pi-meta{font-size:10px;color:#888;margin-top:7px}.pi-compose{display:grid;grid-template-columns:1fr 90px;gap:8px}.pi-compose textarea{resize:vertical;min-height:92px;padding:11px;background:#151515}.pi-compose button{font-weight:750}.pi-busy{color:#ffd166}@media(max-width:700px){.pi-chat{height:500px}.pi-msg{max-width:96%}.pi-compose{grid-template-columns:1fr}}
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
window.addEventListener('load',()=>{const p=document.getElementById('piPrompt');if(p)p.addEventListener('keydown',e=>{if(e.key==='Enter'&&e.ctrlKey){e.preventDefault();sendPi()}});refreshPiStatus();loadPiWorkers();loadPiBench()});
</script>
'''
    dashboard = dashboard.replace("</body>", js + "\n</body>", 1)
    base.DASHBOARD = dashboard

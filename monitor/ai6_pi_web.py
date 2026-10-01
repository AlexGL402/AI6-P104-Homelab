#!/usr/bin/env python3
"""Web UI/API for the local Pi coding agent."""

import json
import os
import subprocess
import time
from pathlib import Path

from pydantic import BaseModel, Field
import ai6_monitor_dynamic as dynamic

base = dynamic.base

REPO = Path.home() / "AI6-P104-Homelab"
PI = Path("/usr/bin/pi")
MODELS = {
    "8b": ("ai6-ollama", "qwen3:8b", "Qwen3 8B"),
    "14b": ("ai6-ollama", "qwen3:14b", "Qwen3 14B"),
}


class PiChatCommand(BaseModel):
    prompt: str = Field(min_length=1, max_length=12000)
    model: str = Field(default="14b", max_length=16)
    history: list[dict] = Field(default_factory=list)


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


@base.app.post("/api/pi/chat")
def pi_chat(cmd: PiChatCommand):
    if cmd.model not in MODELS:
        raise base.HTTPException(status_code=400, detail="unknown Pi model")
    if not PI.is_file():
        raise base.HTTPException(status_code=409, detail=f"Pi not found: {PI}")
    provider, model_id, label = MODELS[cmd.model]
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

    response = response_parts[-1] if response_parts else "(no assistant text)"
    inp = usage.get("input") or usage.get("inputTokens") or usage.get("promptTokens") or 0
    out = usage.get("output") or usage.get("outputTokens") or usage.get("completionTokens") or 0
    total = usage.get("totalTokens") or ((inp or 0) + (out or 0))
    return {
        "ok": True,
        "model": model_id,
        "label": label,
        "elapsed_s": elapsed,
        "response": response.strip(),
        "usage": {"input": inp, "output": out, "total": total},
        "avg_output_tok_s": round(out / elapsed, 2) if out and elapsed else None,
        "tool_calls": tool_calls,
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
        <button onclick="clearPiChat()">New chat</button>
        <button onclick="openPiTerminal()">Terminal</button>
      </div>
    </div>
    <div id="piStatus" class="muted">checking Pi…</div>
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
        r'''.pi-toolbar{display:flex;gap:7px;flex-wrap:wrap}.pi-toolbar select{min-width:190px}.pi-chat{height:590px;overflow:auto;background:#101010;border:1px solid #333;border-radius:12px;padding:14px;margin:12px 0}.pi-empty{color:#777;text-align:center;padding:70px 10px}.pi-msg{max-width:88%;margin:9px 0;padding:10px 12px;border-radius:11px;white-space:pre-wrap;line-height:1.45}.pi-user{margin-left:auto;background:#263044;border:1px solid #3b4b68}.pi-assistant{margin-right:auto;background:#181818;border:1px solid #333}.pi-meta{font-size:10px;color:#888;margin-top:7px}.pi-compose{display:grid;grid-template-columns:1fr 90px;gap:8px}.pi-compose textarea{resize:vertical;min-height:92px;padding:11px;background:#151515}.pi-compose button{font-weight:750}.pi-busy{color:#ffd166}@media(max-width:700px){.pi-chat{height:500px}.pi-msg{max-width:96%}.pi-compose{grid-template-columns:1fr}}
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
async function refreshPiStatus(){
 try{const r=await fetch('/api/pi/status');const s=await r.json();const installed=(s.models||[]).filter(x=>x.installed).map(x=>x.label).join(', ');piStatus.textContent=(s.pi_installed?'● Pi ready':'● Pi missing')+' • '+s.repo+' • models: '+(installed||'none');piStatus.className=s.pi_installed?'status ready':'status down'}catch(e){piStatus.textContent='Pi status error: '+e.message}
}
async function sendPi(){
 const input=document.getElementById('piPrompt'),btn=document.getElementById('piSend'),model=document.getElementById('piModel').value;const prompt=input.value.trim();if(!prompt)return;
 const prior=piHistory.map(x=>({role:x.role,content:x.content}));
 piHistory.push({role:'user',content:prompt});renderPi();input.value='';btn.disabled=true;piStatus.textContent='● Pi working…';piStatus.className='status loading';
 try{
  const r=await fetch('/api/pi/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt,model,history:prior})});const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  const u=d.usage||{};const speed=d.avg_output_tok_s!=null?d.avg_output_tok_s+' agent tok/s':'tok/s —';const toks=(u.input||0)+' in / '+(u.output||0)+' out';const tools=(d.tool_calls||0)+' tools';piHistory.push({role:'assistant',content:d.response||'(no text)',meta:d.label+' • '+d.elapsed_s+' s • '+toks+' • '+speed+' • '+tools});renderPi();piStatus.textContent='● Pi ready • '+d.label+' • '+toks+' • '+speed;piStatus.className='status ready';
 }catch(e){piHistory.push({role:'assistant',content:'ERROR: '+e.message,meta:'request failed'});renderPi();piStatus.textContent='● Pi error';piStatus.className='status error'}
 finally{btn.disabled=false;input.focus()}
}
function clearPiChat(){piHistory=[];renderPi();document.getElementById('piPrompt').focus()}
function openPiTerminal(){window.open(location.protocol+'//'+location.hostname+':8093/','_blank','noopener')}
window.addEventListener('load',()=>{const p=document.getElementById('piPrompt');if(p)p.addEventListener('keydown',e=>{if(e.key==='Enter'&&e.ctrlKey){e.preventDefault();sendPi()}});refreshPiStatus()});
</script>
'''
    dashboard = dashboard.replace("</body>", js + "\n</body>", 1)
    base.DASHBOARD = dashboard

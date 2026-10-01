#!/usr/bin/env python3

import json
import os
import signal
import subprocess
from pathlib import Path

from pydantic import BaseModel
import ai6_monitor_dynamic as dynamic

base = dynamic.base

SOUP_ROOT = Path("/mnt/nvme/data/soup")
DATA_ROOT = Path("/mnt/nvme/data")
PYTHON = SOUP_ROOT / ".venv/bin/python"
HARNESS = SOUP_ROOT / "benchmarks/harness/issue361_nf4_throughput.py"

WEIGHTS = DATA_ROOT / "models/llama31-8b-unsloth"
SHARDS = DATA_ROOT / "models/soup-shards/llama31-8b-unsloth-nf4"

STATE = Path.home() / ".local/state/ai6-monitor/soup"
PIDFILE = STATE / "soup.pid"
LOGFILE = STATE / "soup-live.log"

BASELINE_JSON = (
    Path.home()
    / "AI6-P104-Homelab/benchmarks/huanan-cmp40-soup/"
      "cmp40-stock-gen1x16-unsloth.json"
)

class SoupAction(BaseModel):
    action: str

def _pid():
    try:
        p = int(PIDFILE.read_text().strip())
        os.kill(p, 0)
        return p
    except Exception:
        return None

def _baseline():
    try:
        return json.loads(BASELINE_JSON.read_text())
    except Exception:
        return {
            "tokens_per_s": 174.67,
            "step_time_ms": {"median": 2931.6},
        }

def _tail(path, n=100):
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except Exception:
        return ""

def _live_result():
    text = _tail(LOGFILE, 200)
    out = {}
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("tok/s:"):
            try:
                out["tokens_per_s"] = float(s.split(":",1)[1].strip())
            except Exception:
                pass
        elif s.startswith("step time:") and "median" in s:
            try:
                part = s.split("median",1)[1].split("ms",1)[0].strip()
                out["step_ms"] = float(part)
            except Exception:
                pass
    return out

@base.app.get("/api/soup/status")
def soup_status():
    pid = _pid()
    baseline = _baseline()
    live = _live_result()
    return {
        "running": pid is not None,
        "pid": pid,
        "root": str(SOUP_ROOT),
        "weights": str(WEIGHTS),
        "shards": str(SHARDS),
        "baseline": {
            "tokens_per_s": baseline.get("tokens_per_s", 174.67),
            "step_ms": (baseline.get("step_time_ms") or {}).get("median", 2931.6),
        },
        "live": live,
        "log": _tail(LOGFILE, 80),
    }

@base.app.post("/api/soup/action")
def soup_action(cmd: SoupAction):
    STATE.mkdir(parents=True, exist_ok=True)

    if cmd.action == "stop":
        p = _pid()
        if p:
            try:
                os.kill(p, signal.SIGTERM)
            except ProcessLookupError:
                pass
        PIDFILE.unlink(missing_ok=True)
        return {"ok": True}

    if cmd.action != "run":
        return {"ok": False, "error": "unknown action"}

    if _pid():
        return {"ok": False, "error": "Soup is already running"}

    command = [
        str(PYTHON),
        str(HARNESS),
        "--weights", str(WEIGHTS),
        "--shards", str(SHARDS),
        "--quant", "nf4",
        "--seq", "512",
        "--batch", "1",
        "--warmup", "10",
        "--steps", "50",
        "--buffers", "2",
        "--lora-r", "16",
        "--targets", "q_proj,v_proj",
    ]

    log = LOGFILE.open("w")
    proc = subprocess.Popen(
        command,
        cwd=str(DATA_ROOT),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={
            **os.environ,
            "TMPDIR": str(DATA_ROOT / "tmp"),
        },
    )
    PIDFILE.write_text(str(proc.pid))
    return {"ok": True, "pid": proc.pid}

def install():
    dashboard = base.DASHBOARD

    dashboard = dashboard.replace(
        '<button id="tabMinerBtn" class="tab-btn" onclick="showTopTab(\'miner\')">Miner</button>',
        '<button id="tabMinerBtn" class="tab-btn" onclick="showTopTab(\'miner\')">Miner</button>\n'
        '<button id="tabSoupBtn" class="tab-btn" onclick="showTopTab(\'soup\')">Soup</button>\n'
        '<button id="tabPiBtn" class="tab-btn" onclick="showTopTab(\'pi\')">Pi Agent</button>',
        1,
    )

    html = r'''
<div id="soupTab" style="display:none">
  <div class="section">
    <div class="section-head">
      <div>
        <div class="section-title">Soup / CMP40 Training</div>
        <div class="section-sub">NF4 streamed LoRA benchmark • Llama 3.1 8B • seq 512 • batch 1</div>
      </div>
      <div id="soupState" class="status down">● IDLE</div>
    </div>

    <div class="worker-grid">
      <div class="worker-card">
        <div class="worker-model">Current run</div>
        <div class="worker-stats">
          <div class="worker-stat">Throughput<b id="soupTps">—</b></div>
          <div class="worker-stat">Step median<b id="soupStep">—</b></div>
          <div class="worker-stat">PID<b id="soupPid">—</b></div>
          <div class="worker-stat">PCIe baseline<b>Gen1 x16</b></div>
        </div>
      </div>

      <div class="worker-card">
        <div class="worker-model">Saved baseline</div>
        <div class="worker-stats">
          <div class="worker-stat">Throughput<b id="soupBaseTps">174.67 tok/s</b></div>
          <div class="worker-stat">Step median<b id="soupBaseStep">2931.6 ms</b></div>
          <div class="worker-stat">Quant<b>NF4</b></div>
          <div class="worker-stat">Model<b>Llama 3.1 8B</b></div>
        </div>
      </div>
    </div>

    <div class="worker-actions" style="margin-top:12px">
      <button onclick="soupAction('run')">Run benchmark</button>
      <button onclick="soupAction('stop')">Stop</button>
      <button onclick="refreshSoup()">Refresh</button>
      <span id="soupMsg" class="muted"></span>
    </div>

    <pre id="soupLog" style="margin-top:12px;max-height:460px;overflow:auto;background:#111;border:1px solid #333;border-radius:10px;padding:12px;white-space:pre-wrap"></pre>
  </div>
</div>

<div id="piTab" style="display:none">
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
</div>
'''

    if "</body>" not in dashboard:
        raise RuntimeError("Soup/Pi tab injection failed")

    dashboard = dashboard.replace("</body>", html + "\n</body>", 1)

    js = r'''
<script>
function piAgentUrl(){
  return location.protocol+'//'+location.hostname+':8093/';
}

async function refreshSoup(){
  try{
    const r=await fetch('/api/soup/status');
    const s=await r.json();

    soupState.textContent=s.running?'● RUNNING':'● IDLE';
    soupState.className='status '+(s.running?'ready':'down');
    soupPid.textContent=s.pid??'—';

    soupTps.textContent=s.live.tokens_per_s!=null
      ? s.live.tokens_per_s.toFixed(2)+' tok/s':'—';

    soupStep.textContent=s.live.step_ms!=null
      ? s.live.step_ms.toFixed(1)+' ms':'—';

    soupBaseTps.textContent=(s.baseline.tokens_per_s??174.67).toFixed(2)+' tok/s';
    soupBaseStep.textContent=(s.baseline.step_ms??2931.6).toFixed(1)+' ms';
    soupLog.textContent=s.log||'No log yet.';
  }catch(e){
    soupMsg.textContent='ERROR: '+e.message;
  }
}

async function soupAction(action){
  soupMsg.textContent=action+'...';
  const r=await fetch('/api/soup/action',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action})
  });
  const x=await r.json();
  soupMsg.textContent=r.ok&&x.ok?'OK':(x.error||'ERROR');
  setTimeout(refreshSoup,800);
}
</script>
'''

    dashboard = dashboard.replace("</body>", js + "\n</body>", 1)
    base.DASHBOARD = dashboard

#!/usr/bin/env python3
"""ForgeMiner / PearlHash tab for the AI6 Host Monitor."""

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path

import psutil
from pydantic import BaseModel, Field
from fastapi.responses import PlainTextResponse

import ai6_monitor_dynamic as dynamic

base = dynamic.base

_STATE_DIR = Path(os.environ.get("AI6_MONITOR_STATE_DIR", str(Path.home() / ".local/state/ai6-monitor")))
_MINER_CFG = Path(os.environ.get("AI6_MINER_CONFIG", str(_STATE_DIR / "miner-config.json")))
_MINER_LOG = Path(os.environ.get("AI6_MINER_LOG", str(_STATE_DIR / "forge-miner.log")))
_MINER_PID = Path(os.environ.get("AI6_MINER_PID", str(_STATE_DIR / "forge-miner.pid")))

_DEFAULT_CFG = {
    "name": "Pearl / PearlHash",
    "algorithm": "pearlhash",
    "pool": "prl.kryptex.network:7048",
    "wallet": "",
    "worker": "ai6-cmp40",
    "gpu": "0",
    "binary": "",
    "extra_args": "",
    "temp_limit": 80,
    "temp_resume": 70,
}


class MinerConfigCommand(BaseModel):
    name: str = Field(default="Pearl / PearlHash", max_length=80)
    algorithm: str = Field(default="pearlhash", max_length=40)
    pool: str = Field(default="prl.kryptex.network:7048", max_length=240)
    wallet: str = Field(default="", max_length=240)
    worker: str = Field(default="ai6-cmp40", max_length=120)
    gpu: str = Field(default="0", max_length=100)
    binary: str = Field(default="", max_length=500)
    extra_args: str = Field(default="", max_length=1000)
    temp_limit: int = Field(default=80, ge=40, le=110)
    temp_resume: int = Field(default=70, ge=30, le=100)


class MinerActionCommand(BaseModel):
    action: str


def _load_cfg():
    cfg = dict(_DEFAULT_CFG)
    try:
        if _MINER_CFG.is_file():
            data = json.loads(_MINER_CFG.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                cfg.update({k: v for k, v in data.items() if k in cfg})
    except Exception:
        pass
    return cfg


def _save_cfg(cfg):
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    _MINER_CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def _detect_binary(cfg=None):
    cfg = cfg or _load_cfg()
    explicit = str(cfg.get("binary") or "").strip()
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    which = shutil.which("forge")
    if which:
        candidates.append(Path(which))
    home = Path.home()
    candidates += [
        home / "ForgeMiner" / "forge",
        home / "forgeminer" / "forge",
        home / "forge" / "forge",
        home / "miner" / "forge",
        home / "miners" / "forge" / "forge",
        Path("/opt/ForgeMiner/forge"),
        Path("/opt/forge/forge"),
        Path("/usr/local/bin/forge"),
    ]
    seen = set()
    for p in candidates:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        try:
            if p.is_file() and os.access(p, os.X_OK):
                return str(p)
        except Exception:
            pass
    return explicit or None


def _miner_proc():
    # Prefer the PID we created.
    try:
        if _MINER_PID.is_file():
            pid = int(_MINER_PID.read_text().strip())
            p = psutil.Process(pid)
            if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                return p
    except Exception:
        pass

    # Fallback: detect a forge process started elsewhere.
    for p in psutil.process_iter(["pid", "cmdline", "name", "create_time"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or [])
            name = str(p.info.get("name") or "")
            if re.search(r"(^|/|\s)forge(?:\s|$)", cmd, re.I) or name.lower() == "forge":
                return p
        except Exception:
            continue
    return None


def _vllm_busy():
    for p in psutil.process_iter(["cmdline"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or []).lower()
            if "vllm" in cmd and " serve " in (" " + cmd + " "):
                return True
        except Exception:
            pass
    return False


def _tail_log(max_bytes=120000):
    try:
        if not _MINER_LOG.is_file():
            return ""
        with _MINER_LOG.open("rb") as f:
            size = f.seek(0, os.SEEK_END)
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _parse_miner_log(text):
    clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text or "")
    lines = clean.splitlines()

    rate = None
    rate_unit = None
    pool_rate = None
    pool_rate_unit = None
    accepted = 0
    stale = 0
    rejected = 0
    pool_latency_ms = None
    avg_1m = None
    avg_1h = None
    avg_24h = None

    # Forge TUI table rows look like:
    # | 0  CMP 40HX 8G  40.47 TH/s  78.16 TH/s  3 / 0 / 0 |
    # The first rate is the GPU hashrate; the second is the pool-side rate.
    # Read the newest GPU0 row so the dashboard reflects the miner's latest TUI.
    gpu_row_re = re.compile(
        r"\|\s*0\s+.+?\s+([0-9]+(?:\.[0-9]+)?)\s*(TH/s|GH/s|MH/s|kH/s|H/s)"
        r"\s+([0-9]+(?:\.[0-9]+)?)\s*(TH/s|GH/s|MH/s|kH/s|H/s)"
        r"\s+(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*\|",
        re.I,
    )
    for line in reversed(lines):
        m = gpu_row_re.search(line)
        if m:
            rate = float(m.group(1))
            rate_unit = m.group(2)
            pool_rate = float(m.group(3))
            pool_rate_unit = m.group(4)
            accepted = int(m.group(5))
            stale = int(m.group(6))
            rejected = int(m.group(7))
            break

    # Session Stats gives stable averages and latency; use the newest values.
    for line in reversed(lines):
        if pool_latency_ms is None:
            m = re.search(r"\|\s*Latency\s+~?\s*(\d+)\s*ms", line, re.I)
            if m:
                pool_latency_ms = int(m.group(1))
        if avg_1m is None:
            m = re.search(r"\|\s*Avg\s+1\s+min\s+([0-9]+(?:\.[0-9]+)?)\s*(TH/s|GH/s|MH/s|kH/s|H/s)", line, re.I)
            if m:
                avg_1m = float(m.group(1))
        if avg_1h is None:
            m = re.search(r"\|\s*Avg\s+1\s+hr\s+([0-9]+(?:\.[0-9]+)?)\s*(TH/s|GH/s|MH/s|kH/s|H/s)", line, re.I)
            if m:
                avg_1h = float(m.group(1))
        if avg_24h is None:
            m = re.search(r"\|\s*Avg\s+24\s+hr\s+([0-9]+(?:\.[0-9]+)?)\s*(TH/s|GH/s|MH/s|kH/s|H/s)", line, re.I)
            if m:
                avg_24h = float(m.group(1))
        if all(v is not None for v in (pool_latency_ms, avg_1m, avg_1h, avg_24h)):
            break

    # Fallback for very early startup before the first TUI table is printed.
    if rate is None:
        for line in reversed(lines):
            matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*(TH/s|GH/s|MH/s|kH/s|H/s)", line, re.I)
            if matches:
                val, unit = matches[0]
                rate = float(val)
                rate_unit = unit
                break

    return {
        "hashrate": rate,
        "hashrate_unit": rate_unit,
        "pool_hashrate": pool_rate,
        "pool_hashrate_unit": pool_rate_unit,
        "avg_1m": avg_1m,
        "avg_1h": avg_1h,
        "avg_24h": avg_24h,
        "accepted": accepted,
        "stale": stale,
        "rejected": rejected,
        "latency_ms": pool_latency_ms,
    }


def _gpu_snapshot():
    out = []
    try:
        p = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,power.draw,power.limit,temperature.gpu,fan.speed,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=3.0,
        )
        if p.returncode != 0:
            return out
        for line in p.stdout.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 9:
                continue
            def num(v):
                try:
                    return float(v)
                except Exception:
                    return None
            out.append({
                "index": int(float(parts[0])),
                "name": parts[1],
                "load_pct": num(parts[2]),
                "power_w": num(parts[3]),
                "power_limit_w": num(parts[4]),
                "temp_c": num(parts[5]),
                "fan_pct": num(parts[6]),
                "vram_used_mib": num(parts[7]),
                "vram_total_mib": num(parts[8]),
            })
    except Exception:
        pass
    return out


def _status():
    cfg = _load_cfg()
    proc = _miner_proc()
    running = proc is not None
    uptime = None
    pid = None
    if proc:
        try:
            pid = proc.pid
            uptime = max(0, time.time() - proc.create_time())
        except Exception:
            pass
    log = _tail_log()
    parsed = _parse_miner_log(log)
    binary = _detect_binary(cfg)
    return {
        "running": running,
        "pid": pid,
        "uptime_s": uptime,
        "vllm_busy": _vllm_busy(),
        "binary_detected": binary,
        "config": cfg,
        "metrics": parsed,
        "gpus": _gpu_snapshot(),
        "log_path": str(_MINER_LOG),
    }


@base.app.get("/api/miner/status")
def api_miner_status():
    return _status()


@base.app.post("/api/miner/config")
def api_miner_config(cmd: MinerConfigCommand):
    cfg = cmd.model_dump()
    _save_cfg(cfg)
    return {"ok": True, "config": cfg, "binary_detected": _detect_binary(cfg)}


@base.app.post("/api/miner/control")
def api_miner_control(cmd: MinerActionCommand):
    action = (cmd.action or "").strip().lower()

    if action == "start":
        if _miner_proc():
            return {"ok": True, "message": "Miner is already running."}
        if _vllm_busy():
            raise base.HTTPException(status_code=409, detail="vLLM is running. Stop AI inference before starting the miner.")

        cfg = _load_cfg()
        binary = _detect_binary(cfg)
        if not binary:
            raise base.HTTPException(
                status_code=400,
                detail="ForgeMiner binary not found. Set the Miner binary field (path to forge) and Save config.",
            )
        if not str(cfg.get("wallet") or "").strip():
            raise base.HTTPException(status_code=400, detail="Wallet is empty. Enter the PRL wallet and Save config.")
        if not str(cfg.get("pool") or "").strip():
            raise base.HTTPException(status_code=400, detail="Pool is empty.")

        argv = [
            binary,
            "--algorithm", str(cfg.get("algorithm") or "pearlhash"),
            "--wallet", str(cfg["wallet"]),
            "--pool", str(cfg["pool"]),
            "--worker", str(cfg.get("worker") or "ai6"),
        ]
        gpu = str(cfg.get("gpu") or "").strip()
        if gpu:
            argv += ["--gpu", gpu]
        argv += ["--temp-limit", str(cfg.get("temp_limit") or 80)]
        argv += ["--temp-resume", str(cfg.get("temp_resume") or 70)]
        extra = str(cfg.get("extra_args") or "").strip()
        if extra:
            try:
                argv += shlex.split(extra)
            except ValueError as e:
                raise base.HTTPException(status_code=400, detail=f"Invalid extra args: {e}")

        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        logf = _MINER_LOG.open("ab", buffering=0)
        header = ("\n\n=== AI6 monitor start " + time.strftime("%Y-%m-%d %H:%M:%S %z") + " ===\n").encode()
        logf.write(header)
        proc = subprocess.Popen(
            argv,
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=str(Path(binary).parent),
            start_new_session=True,
            env=os.environ.copy(),
        )
        logf.close()
        _MINER_PID.write_text(str(proc.pid), encoding="utf-8")
        return {"ok": True, "message": "Miner started.", "pid": proc.pid}

    if action == "stop":
        proc = _miner_proc()
        if not proc:
            try:
                _MINER_PID.unlink(missing_ok=True)
            except Exception:
                pass
            return {"ok": True, "message": "Miner is not running."}
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        try:
            _MINER_PID.unlink(missing_ok=True)
        except Exception:
            pass
        return {"ok": True, "message": "Miner stopped."}

    raise base.HTTPException(status_code=400, detail="action must be start or stop")


@base.app.get("/api/miner/logs")
def api_miner_logs():
    return PlainTextResponse(_tail_log(), media_type="text/plain; charset=utf-8")


def install():
    dashboard = base.DASHBOARD

    # Add a top-level Miner tab next to Benchs.
    dashboard = dashboard.replace(
        '<button id="tabBenchsBtn" class="tab-btn" onclick="showTopTab(\'benchs\')">Benchs</button>',
        '<button id="tabBenchsBtn" class="tab-btn" onclick="showTopTab(\'benchs\')">Benchs</button>\n'
        '<button id="tabMinerBtn" class="tab-btn" onclick="showTopTab(\'miner\')">Miner</button>',
        1,
    )

    miner_html = r'''
<div id="minerTab" style="display:none">
  <div class="section">
    <div class="section-head">
      <div>
        <div class="section-title">Pearl / ForgeMiner</div>
        <div class="section-sub">Mine on idle GPUs. Start is blocked while vLLM is running.</div>
      </div>
      <div id="minerState" class="status down">● STOPPED</div>
    </div>

    <div class="miner-summary">
      <div class="miner-kpi"><span>Hashrate</span><b id="minerHashrate">—</b><small id="minerAlgo">PearlHash</small></div>
      <div class="miner-kpi"><span>Shares</span><b id="minerShares">—</b><small id="minerLatency">accepted / rejected</small></div>
      <div class="miner-kpi"><span>Uptime</span><b id="minerUptime">—</b><small id="minerPid">PID —</small></div>
      <div class="miner-kpi"><span>AI status</span><b id="minerAiState">—</b><small>miner start interlock</small></div>
    </div>

    <div class="miner-config">
      <label>Name<input id="minerName" value="Pearl / PearlHash"></label>
      <label>Pool<input id="minerPool" value="prl.kryptex.network:7048"></label>
      <label>Wallet<input id="minerWallet" placeholder="PRL wallet"></label>
      <label>Worker<input id="minerWorker" value="ai6-cmp40"></label>
      <label>Algorithm<input id="minerAlgorithm" value="pearlhash"></label>
      <label>GPU(s)<input id="minerGpu" value="0" placeholder="0 or 0,1"></label>
      <label>Miner binary<input id="minerBinary" placeholder="/path/to/forge"></label>
      <label>Extra args<input id="minerExtra" placeholder="e.g. --cmp-install"></label>
      <label>Temp limit °C<input id="minerTempLimit" type="number" value="80" min="40" max="110"></label>
      <label>Resume °C<input id="minerTempResume" type="number" value="70" min="30" max="100"></label>
    </div>

    <div class="miner-actions">
      <button type="button" onclick="saveMinerConfig()">Save config</button>
      <button type="button" class="miner-start" onclick="controlMiner('start')">Start</button>
      <button type="button" class="miner-stop" onclick="controlMiner('stop')">Stop</button>
      <span id="minerMsg" class="muted"></span>
    </div>

    <div class="miner-power-row">
      <label>GPU power limit
        <select id="minerPlPreset">
          <option value="125">125 W</option>
          <option value="130" selected>130 W</option>
          <option value="140">140 W</option>
          <option value="150">150 W</option>
          <option value="165">165 W</option>
          <option value="184">184 W</option>
          <option value="200">200 W</option>
          <option value="220">220 W</option>
        </select>
      </label>
      <label>GPU index<input id="minerPlGpu" type="number" value="0" min="0" max="31"></label>
      <button type="button" onclick="setMinerPowerLimit()">Set PL</button>
      <span id="minerPlMsg" class="muted"></span>
    </div>

    <div id="minerGpuGrid" class="vllm-gpu-grid"></div>

    <div class="miner-log-wrap">
      <div class="miner-log-head"><span id="minerLogPath">ForgeMiner log</span><button type="button" onclick="refreshMinerLogs()">Refresh log</button></div>
      <pre id="minerLogText">No miner log loaded yet.</pre>
    </div>
  </div>
</div>
'''
    if "</body>" not in dashboard:
        raise RuntimeError("Miner tab injection failed: </body> marker not found")
    if 'id="minerTab"' not in dashboard:
        dashboard = dashboard.replace("</body>", miner_html + "\n</body>", 1)

    css = r'''
.miner-summary{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:8px;margin:10px 0}
.miner-kpi{background:#171717;border:1px solid #303030;border-radius:9px;padding:9px 11px}
.miner-kpi span,.miner-kpi small{display:block;color:#92979c;font-size:10px}.miner-kpi b{display:block;color:#eee;font-size:21px;margin:2px 0}
.miner-config{display:grid;grid-template-columns:repeat(5,minmax(150px,1fr));gap:8px;align-items:end}
.miner-config label,.miner-power-row label{font-size:10px;color:#aaa}.miner-config input,.miner-power-row input,.miner-power-row select{display:block;width:100%;margin-top:4px;padding:7px;background:#111;color:#ddd;border:1px solid #3a3a3a;border-radius:6px}
.miner-actions,.miner-power-row{display:flex;gap:7px;align-items:end;flex-wrap:wrap;margin-top:10px}.miner-power-row label{min-width:120px}.miner-power-row button{height:32px}
.miner-start{border-color:#2f7540!important;color:#72e28a!important;background:#132519!important}.miner-stop{border-color:#7a3434!important;color:#ff8585!important;background:#2a1515!important}
.miner-log-wrap{margin-top:10px;background:#101010;border:1px solid #303030;border-radius:8px;padding:8px}.miner-log-head{display:flex;justify-content:space-between;align-items:center;color:#888;font-size:10px}.miner-log-head button{padding:4px 8px;font-size:10px}.miner-log-wrap pre{margin:7px 0 0;max-height:330px;overflow:auto;white-space:pre-wrap;word-break:break-word;font-size:10px;line-height:1.35;color:#cfcfcf}
@media(max-width:1100px){.miner-config{grid-template-columns:repeat(2,minmax(150px,1fr))}.miner-summary{grid-template-columns:repeat(2,minmax(150px,1fr))}}
'''
    dashboard = dashboard.replace("</style>", css + "\n</style>", 1)

    # Tab switching is finalized centrally in ai6_monitor_auto.py.

    js = r'''
let minerConfigLoaded=false;

function minerFmtUptime(s){
 if(s==null)return '—';
 s=Math.floor(s);const d=Math.floor(s/86400);s%=86400;const h=Math.floor(s/3600);s%=3600;const m=Math.floor(s/60);
 return (d?d+'d ':'')+(h?h+'h ':'')+m+'m';
}

function loadMinerConfig(c,detected){
 if(minerConfigLoaded)return;
 minerName.value=c.name||'Pearl / PearlHash';
 minerPool.value=c.pool||'prl.kryptex.network:7048';
 minerWallet.value=c.wallet||'';
 minerWorker.value=c.worker||'ai6-cmp40';
 minerAlgorithm.value=c.algorithm||'pearlhash';
 minerGpu.value=c.gpu||'0';
 minerBinary.value=c.binary||detected||'';
 minerExtra.value=c.extra_args||'';
 minerTempLimit.value=c.temp_limit||80;
 minerTempResume.value=c.temp_resume||70;
 minerConfigLoaded=true;
}

async function refreshMiner(){
 try{
  const r=await fetch('/api/miner/status',{cache:'no-store'});const s=await r.json();
  if(!r.ok)throw new Error(s.detail||JSON.stringify(s));
  loadMinerConfig(s.config||{},s.binary_detected);
  minerState.textContent=s.running?'● RUNNING':'● STOPPED';
  minerState.className='status '+(s.running?'ready':'down');
  const m=s.metrics||{};
  minerHashrate.textContent=m.hashrate==null?'—':Number(m.hashrate).toFixed(2)+' '+(m.hashrate_unit||'');
  minerAlgo.textContent=(m.avg_1m==null?'':'avg 1m '+Number(m.avg_1m).toFixed(2)+' TH/s • ')+((s.config&&s.config.algorithm)||'pearlhash');
  minerShares.textContent=(m.accepted||0)+' / '+(m.stale||0)+' / '+(m.rejected||0);
  minerLatency.textContent=(m.latency_ms==null?'A / S / R':'A / S / R • '+m.latency_ms+' ms')+(m.pool_hashrate==null?'':' • pool '+Number(m.pool_hashrate).toFixed(2)+' '+(m.pool_hashrate_unit||''));
  minerUptime.textContent=minerFmtUptime(s.uptime_s);
  minerPid.textContent='PID '+(s.pid??'—');
  minerAiState.textContent=s.vllm_busy?'AI BUSY':'IDLE';
  minerAiState.className=s.vllm_busy?'bad':'ok';
  minerLogPath.textContent=s.log_path||'ForgeMiner log';
  renderMinerGpuGrid(s.gpus||[],String((s.config||{}).gpu||'0'));
 }catch(e){minerMsg.textContent='Status error: '+e.message;}
}

function renderMinerGpuGrid(gpus,gpuString){
 const sel=new Set(String(gpuString||'').split(',').map(x=>Number(x.trim())).filter(Number.isFinite));
 minerGpuGrid.innerHTML=(gpus||[]).map(g=>{
  const active=sel.has(g.index);
  const load=Number(g.load_pct||0),p=Number(g.power_w||0),pl=Number(g.power_limit_w||0),t=Number(g.temp_c||0);
  const loadCls=load>=90?'ok':load>=50?'warn':'';
  const pCls=pl&&p/pl>=.98?'bad':pl&&p/pl>=.90?'warn':'ok';
  const tCls=t>=80?'bad':t>=70?'warn':'ok';
  const used=Number(g.vram_used_mib||0)/1024,total=Number(g.vram_total_mib||0)/1024;
  return '<div class="vllm-gpu-card'+(active?' selected':'')+'">'+
   '<div class="vllm-gpu-card-head"><b>#'+g.index+' '+escHtml(g.name||'GPU')+'</b><span>'+(active?'miner selected':'available')+'</span></div>'+
   '<div class="vllm-gpu-stats">'+
    '<div>Load<b class="'+loadCls+'">'+load.toFixed(0)+'%</b></div>'+
    '<div>Power<b class="'+pCls+'">'+p.toFixed(1)+' W</b><span> / '+pl.toFixed(0)+' W</span></div>'+
    '<div>Temp<b class="'+tCls+'">'+t.toFixed(0)+'°C</b><span> fan '+Number(g.fan_pct||0).toFixed(0)+'%</span></div>'+
    '<div>VRAM<b>'+used.toFixed(2)+' / '+total.toFixed(2)+' GiB</b></div>'+
   '</div></div>';
 }).join('')||'<div class="muted">No NVIDIA GPUs detected</div>';
}

async function saveMinerConfig(){
 const payload={
  name:minerName.value.trim(),algorithm:minerAlgorithm.value.trim(),pool:minerPool.value.trim(),
  wallet:minerWallet.value.trim(),worker:minerWorker.value.trim(),gpu:minerGpu.value.trim(),
  binary:minerBinary.value.trim(),extra_args:minerExtra.value.trim(),
  temp_limit:Number(minerTempLimit.value),temp_resume:Number(minerTempResume.value)
 };
 minerMsg.textContent='Saving…';
 try{
  const r=await fetch('/api/miner/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  minerMsg.textContent='Config saved'+(d.binary_detected?' • '+d.binary_detected:'');
 }catch(e){minerMsg.textContent='Save error: '+e.message;}
}

async function controlMiner(action){
 if(action==='start'){
  await saveMinerConfig();
  if(!confirm('Start ForgeMiner on selected GPU(s)?'))return;
  try{
   await setMinerPowerLimit(true);
  }catch(e){
   minerMsg.textContent='Start blocked: '+e.message;
   return;
  }
 }else if(!confirm('Stop ForgeMiner?'))return;
 minerMsg.textContent=action==='start'?'Starting miner…':'Stopping miner…';
 try{
  const r=await fetch('/api/miner/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})});
  const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  minerMsg.textContent=d.message||'OK';
  setTimeout(refreshMiner,700);setTimeout(refreshMinerLogs,1200);
 }catch(e){minerMsg.textContent='Miner error: '+e.message;}
}

async function setMinerPowerLimit(throwOnError=false){
 const gpu=Number(minerPlGpu.value),watts=Number(minerPlPreset.value);
 minerPlMsg.textContent='Setting '+watts+' W…';
 try{
  const r=await fetch('/api/gpu/power-limit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({gpu,watts})});
  const d=await r.json();if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  minerPlMsg.textContent='GPU'+gpu+' PL '+Number(d.current_w).toFixed(0)+' W';
  await refreshMiner();
  return d;
 }catch(e){
  minerPlMsg.textContent='PL error: '+e.message;
  if(throwOnError)throw e;
  return null;
 }
}

async function refreshMinerLogs(){
 try{
  const r=await fetch('/api/miner/logs',{cache:'no-store'});
  const t=await r.text();
  minerLogText.textContent=t||'(empty)';
  minerLogText.scrollTop=minerLogText.scrollHeight;
 }catch(e){minerLogText.textContent='Log error: '+e.message;}
}

setInterval(()=>{if(document.getElementById('minerTab')&&document.getElementById('minerTab').style.display!=='none')refreshMiner();},2000);
'''
    dashboard = dashboard.replace("refresh();setInterval(refresh,2000);", js + "\nrefresh();setInterval(refresh,2000);", 1)

    base.DASHBOARD = dashboard
    dynamic.base.DASHBOARD = dashboard
    dynamic.app = base.app

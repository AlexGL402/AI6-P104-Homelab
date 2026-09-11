#!/usr/bin/env python3
import csv
import json
import os
import platform
import re
import socket
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import psutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

APP_TITLE = "AI6 Host Monitor"
CSV_PATH = Path(os.environ.get("AI6_MONITOR_CSV", "/var/lib/ai6-monitor/psu-test.csv"))

app = FastAPI(title=APP_TITLE, version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def gpu_stats():
    fields = [
        "index",
        "name",
        "temperature.gpu",
        "utilization.gpu",
        "memory.used",
        "memory.total",
        "power.draw",
        "power.limit",
        "fan.speed",
        "clocks.current.graphics",
        "clocks.current.memory",
    ]
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True)
    except FileNotFoundError:
        raise RuntimeError("nvidia-smi not found on host")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(e.stderr.strip() or "nvidia-smi failed")

    gpus = []
    for line in p.stdout.strip().splitlines():
        cols = [x.strip() for x in line.split(",")]
        if len(cols) != len(fields):
            continue
        fan = _float(cols[8])
        if fan is not None and not (0 <= fan <= 100):
            fan = None
        gpus.append({
            "index": _int(cols[0]),
            "name": cols[1],
            "temperature_c": _float(cols[2]),
            "utilization_pct": _float(cols[3]),
            "memory_used_mib": _float(cols[4]),
            "memory_total_mib": _float(cols[5]),
            "power_w": _float(cols[6]),
            "power_limit_w": _float(cols[7]),
            "fan_pct": fan,
            "graphics_clock_mhz": _float(cols[9]),
            "memory_clock_mhz": _float(cols[10]),
        })
    return gpus


def cpu_temperature():
    try:
        temps = psutil.sensors_temperatures(fahrenheit=False)
    except Exception:
        return None
    values = []
    for entries in temps.values():
        for t in entries:
            if t.current is not None and -20 < t.current < 150:
                values.append(float(t.current))
    return max(values) if values else None


def service_ok(port: int):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.35):
            return True
    except OSError:
        return False


def systemd_active(name: str):
    return subprocess.run(
        ["systemctl", "is-active", "--quiet", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def systemd_enabled(name: str):
    return subprocess.run(
        ["systemctl", "is-enabled", "--quiet", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def worker_profile():
    return "222" if systemd_enabled("llama2-8081") else "33"


def run_workerctl(*args):
    cmd = ["sudo", "/usr/local/sbin/ai6-workerctl", *args]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "workerctl failed").strip())
    return (p.stdout or "").strip()


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("1.1.1.1", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(platform.node())
        except Exception:
            return None


def _health_state(port: int, active: bool):
    if not active:
        return "down", "Down"
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.7) as r:
            data = json.load(r)
        if data.get("status") == "ok":
            return "ready", "Ready"
        return "starting", str(data.get("status") or "Starting")
    except urllib.error.HTTPError as e:
        try:
            data = json.loads(e.read().decode("utf-8", "replace"))
            err = data.get("error") or {}
            msg = str(err.get("message") or data.get("detail") or f"HTTP {e.code}")
        except Exception:
            msg = f"HTTP {e.code}"
        if e.code == 503 and "loading" in msg.lower():
            return "loading", "Loading model"
        return "error", msg
    except Exception:
        return "starting", "Starting"


def worker_details(port: int):
    active = service_ok(port)
    state, status_text = _health_state(port, active)
    info = {
        "active": active,
        "ready": state == "ready",
        "state": state,
        "status_text": status_text,
        "model": None,
        "last_tok_s": None,
        "last_prompt_tok_s": None,
        "last_prompt_tokens": None,
        "uptime_s": None,
    }
    if not active:
        return info

    if state == "ready":
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=0.7) as r:
                data = json.load(r)
            models = data.get("data") or []
            if models:
                info["model"] = models[0].get("id")
        except Exception:
            pass

    service = f"llama2-{port}" if systemd_active(f"llama2-{port}") else f"llama-{port}"
    try:
        p = subprocess.run(
            ["systemctl", "show", service, "--property=ActiveEnterTimestampMonotonic", "--value"],
            capture_output=True, text=True, timeout=1,
        )
        entered = int((p.stdout or "0").strip() or 0)
        if entered:
            boot_us = int((datetime.now().timestamp() - psutil.boot_time()) * 1_000_000)
            info["uptime_s"] = max(0, int((boot_us - entered) / 1_000_000))
    except Exception:
        pass

    try:
        p = subprocess.run(
            ["journalctl", "-u", service, "-n", "120", "--no-pager", "-o", "cat"],
            capture_output=True, text=True, timeout=2,
        )
        lines = (p.stdout or "").splitlines()
        for line in reversed(lines):
            if info["last_tok_s"] is None and "eval time" in line and "tokens per second" in line and "prompt eval time" not in line:
                m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s+tokens per second", line)
                if m:
                    info["last_tok_s"] = float(m.group(1))
            if info["last_prompt_tok_s"] is None and "prompt eval time" in line:
                m = re.search(r"/\s+(\d+)\s+tokens.*?([0-9]+(?:\.[0-9]+)?)\s+tokens per second", line)
                if m:
                    info["last_prompt_tokens"] = int(m.group(1))
                    info["last_prompt_tok_s"] = float(m.group(2))
            if info["last_tok_s"] is not None and info["last_prompt_tok_s"] is not None:
                break
    except Exception:
        pass
    return info


def collect_stats():
    vm = psutil.virtual_memory()
    root = psutil.disk_usage("/")
    net = psutil.net_io_counters()
    gpus = gpu_stats()
    powers = [g["power_w"] for g in gpus if g["power_w"] is not None]
    temps = [g["temperature_c"] for g in gpus if g["temperature_c"] is not None]
    workers = {str(p): worker_details(p) for p in (8081, 8082, 8083)}

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": {
            "hostname": platform.node(),
            "kernel": platform.release(),
            "uptime_s": int(datetime.now().timestamp() - psutil.boot_time()),
            "lan_ip": lan_ip(),
        },
        "cpu": {
            "usage_pct": psutil.cpu_percent(interval=0.15),
            "load_1m": os.getloadavg()[0] if hasattr(os, "getloadavg") else None,
            "load_5m": os.getloadavg()[1] if hasattr(os, "getloadavg") else None,
            "load_15m": os.getloadavg()[2] if hasattr(os, "getloadavg") else None,
            "logical_cpus": psutil.cpu_count(logical=True),
            "physical_cpus": psutil.cpu_count(logical=False),
            "temperature_c": cpu_temperature(),
        },
        "memory": {
            "used_bytes": vm.used,
            "total_bytes": vm.total,
            "available_bytes": vm.available,
            "usage_pct": vm.percent,
        },
        "disk": {
            "used_bytes": root.used,
            "total_bytes": root.total,
            "free_bytes": root.free,
            "usage_pct": root.percent,
        },
        "network": {
            "bytes_sent": net.bytes_sent,
            "bytes_recv": net.bytes_recv,
        },
        "gpu": {
            "count": len(gpus),
            "power_total_w": round(sum(powers), 2),
            "temperature_max_c": max(temps) if temps else None,
            "memory_used_total_mib": round(sum(g["memory_used_mib"] or 0 for g in gpus), 1),
            "memory_total_mib": round(sum(g["memory_total_mib"] or 0 for g in gpus), 1),
            "devices": gpus,
        },
        "llama": {
            "profile": worker_profile(),
            "8081": workers["8081"]["active"],
            "8082": workers["8082"]["active"],
            "8083": workers["8083"]["active"],
            "workers": workers,
        },
    }


class PsuSample(BaseModel):
    voltage_12v: float = Field(..., ge=0, le=20)
    note: str = Field(default="", max_length=200)


class WorkerCommand(BaseModel):
    action: str
    port: int | None = None
    profile: str | None = None


@app.get("/api/stats")
def api_stats():
    try:
        return collect_stats()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/api/psu-sample")
def psu_sample(sample: PsuSample):
    try:
        stats = collect_stats()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not CSV_PATH.exists()
    row = {
        "timestamp": stats["timestamp"],
        "voltage_12v": sample.voltage_12v,
        "gpu_power_total_w": stats["gpu"]["power_total_w"],
        "gpu_temp_max_c": stats["gpu"]["temperature_max_c"],
        "cpu_usage_pct": stats["cpu"]["usage_pct"],
        "ram_usage_pct": stats["memory"]["usage_pct"],
        "note": sample.note,
    }
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if new_file:
            w.writeheader()
        w.writerow(row)
    return {"ok": True, "saved": str(CSV_PATH), "sample": row}


@app.post("/api/workers/action")
def worker_action(cmd: WorkerCommand):
    try:
        if cmd.action in ("start", "stop", "restart"):
            if cmd.port not in (8081, 8082, 8083):
                raise HTTPException(status_code=400, detail="invalid worker port")
            out = run_workerctl(cmd.action, str(cmd.port))
        elif cmd.action == "profile":
            if cmd.profile not in ("33", "222"):
                raise HTTPException(status_code=400, detail="invalid profile")
            out = run_workerctl("profile", cmd.profile)
        else:
            raise HTTPException(status_code=400, detail="invalid action")
        return {"ok": True, "output": out}
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


DASHBOARD = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI6 Host Monitor</title>
<style>
body{font-family:system-ui,Arial,sans-serif;margin:20px;background:#111;color:#eee}h1{margin:0 0 6px}.muted{color:#aaa}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin-top:16px}.card{background:#1b1b1b;border:1px solid #333;border-radius:12px;padding:14px}.big{font-size:28px;font-weight:700}.gpu{display:grid;grid-template-columns:42px 1fr;gap:8px;border-top:1px solid #333;padding:9px 0}.gpu:first-child{border-top:0}.ok{color:#7be495}.warn{color:#ffd166}.bad{color:#ff6b6b}.status{font-weight:700}.ready{color:#7be495}.loading,.starting{color:#ffd166}.down,.error{color:#ff6b6b}input,button{font:inherit;padding:8px;border-radius:8px;border:1px solid #555;background:#222;color:#eee}button{cursor:pointer}.bar{height:8px;background:#333;border-radius:5px;overflow:hidden;margin-top:5px}.fill{height:100%;background:#aaa;width:0%}table{width:100%;border-collapse:collapse}td{padding:4px 2px;border-bottom:1px solid #2d2d2d}</style>
</head><body>
<h1>AI6 Host Monitor</h1><div class="muted" id="stamp">loading...</div>
<div class="grid">
 <div class="card"><div>CPU</div><div class="big" id="cpu">-</div><div id="cpu2" class="muted"></div></div>
 <div class="card"><div>RAM</div><div class="big" id="ram">-</div><div id="ram2" class="muted"></div></div>
 <div class="card"><div>GPU total power</div><div class="big" id="power">-</div><div id="gputemp" class="muted"></div></div>
 <div class="card"><div>Disk /</div><div class="big" id="disk">-</div><div id="disk2" class="muted"></div></div>
 <div class="card"><div>Network</div><div class="big" id="net">-</div><div id="net2" class="muted"></div></div>
 <div class="card"><div>System</div><div class="big" id="uptime">-</div><div id="system2" class="muted"></div></div>
 <div class="card"><div>GPU VRAM</div><div class="big" id="vramtotal">-</div><div id="vramtotal2" class="muted"></div></div>
 <div class="card"><div>llama workers</div><div class="big" id="workers">-</div><div id="profile" class="muted">profile -</div><div id="workerbuttons" style="margin-top:8px"></div><div style="margin-top:8px"><button onclick="setProfile('33')">3+3</button> <button onclick="setProfile('222')">2+2+2</button></div><div id="workermsg" class="muted" style="margin-top:6px"></div></div>
</div>
<div class="grid"><div class="card" style="grid-column:1/-1"><h3>GPUs</h3><div id="gpus"></div></div></div>
<div class="grid"><div class="card"><h3>PSU 12V sample</h3><p class="muted">Enter the multimeter reading. The current GPU power and temperatures will be logged to CSV on the host.</p><input id="v12" type="number" step="0.01" placeholder="12.05"><input id="note" placeholder="note, e.g. 400W"><button onclick="saveSample()">Save sample</button><div id="saved" class="muted"></div></div></div>
<div class="grid"><div class="card" style="grid-column:1/-1"><h3>Web Terminal</h3><p class="muted">Direct shell on the AI6 host. It runs as the normal Linux user and is protected by separate HTTP Basic authentication.</p><button onclick="openTerminal()">Open Web Terminal</button> <button onclick="toggleTerminal()">Show / hide below</button><div id="termwrap" style="display:none;margin-top:12px"><iframe id="termframe" title="AI6 Web Terminal" style="width:100%;height:520px;border:1px solid #333;border-radius:10px;background:#000"></iframe></div></div></div>
<script>
const mib=(v)=>v==null?'?':(v/1024).toFixed(2)+' GiB';
const gib=(v)=>v==null?'?':(v/1073741824).toFixed(2)+' GiB';
const cls=(t)=>t==null?'':(t>=80?'bad':t>=65?'warn':'ok');
const fmtUptime=(s)=>{s=Math.max(0,Math.floor(s||0));const d=Math.floor(s/86400);s%=86400;const h=Math.floor(s/3600);s%=3600;const m=Math.floor(s/60);return (d?d+'d ':'')+h+'h '+m+'m'};
const stateIcon=(w)=>w.state==='ready'?'✅':(w.state==='loading'||w.state==='starting')?'⏳':'❌';
let prevNet=null,prevNetTs=null;
async function refresh(){
 try{const r=await fetch('/api/stats');const s=await r.json();if(!r.ok)throw new Error(JSON.stringify(s));
 document.getElementById('stamp').textContent=s.host.hostname+' • '+new Date(s.timestamp).toLocaleString();
 cpu.textContent=s.cpu.usage_pct.toFixed(1)+'%'; cpu2.textContent='load '+s.cpu.load_1m.toFixed(2)+' / '+s.cpu.load_5m.toFixed(2)+' / '+s.cpu.load_15m.toFixed(2)+(s.cpu.temperature_c!=null?' • '+s.cpu.temperature_c.toFixed(0)+'°C':'');
 ram.textContent=s.memory.usage_pct.toFixed(1)+'%';ram2.textContent=(s.memory.used_bytes/1073741824).toFixed(2)+' / '+(s.memory.total_bytes/1073741824).toFixed(2)+' GiB';
 power.textContent=s.gpu.power_total_w.toFixed(1)+' W';gputemp.innerHTML='max temp <span class="'+cls(s.gpu.temperature_max_c)+'">'+(s.gpu.temperature_max_c??'?')+'°C</span>';
 disk.textContent=s.disk.usage_pct.toFixed(1)+'%';disk2.textContent=gib(s.disk.used_bytes)+' used • '+gib(s.disk.free_bytes)+' free / '+gib(s.disk.total_bytes);
 uptime.textContent=fmtUptime(s.host.uptime_s);system2.textContent=(s.host.lan_ip||location.hostname)+' • kernel '+s.host.kernel;
 vramtotal.textContent=mib(s.gpu.memory_used_total_mib);vramtotal2.textContent='used / '+mib(s.gpu.memory_total_mib)+' total • '+s.gpu.count+' GPUs';
 const now=Date.now()/1000;
 if(prevNet && prevNetTs){const dt=Math.max(.1,now-prevNetTs);const rx=(s.network.bytes_recv-prevNet.rx)*8/dt/1e6;const tx=(s.network.bytes_sent-prevNet.tx)*8/dt/1e6;net.textContent='↓ '+rx.toFixed(2)+' Mbps';net2.textContent='↑ '+tx.toFixed(2)+' Mbps • total ↓ '+gib(s.network.bytes_recv)+' ↑ '+gib(s.network.bytes_sent)}
 else{net.textContent='warming…';net2.textContent='total ↓ '+gib(s.network.bytes_recv)+' ↑ '+gib(s.network.bytes_sent)}
 prevNet={rx:s.network.bytes_recv,tx:s.network.bytes_sent};prevNetTs=now;
 const ports=s.llama.profile==='222'?[8081,8082,8083]:[8081,8082];
 const ws=s.llama.workers||{};
 workers.innerHTML=ports.map(p=>stateIcon(ws[String(p)]||{})).join(' ');
 profile.textContent='profile '+(s.llama.profile==='222'?'2+2+2':'3+3')+' • ✅ Ready  ⏳ Loading  ❌ Down';
 workerbuttons.innerHTML=ports.map(p=>{const w=ws[String(p)]||{};const model=w.model||((w.state==='loading')?'loading model…':'model ?');const speed=w.last_tok_s!=null?w.last_tok_s.toFixed(2)+' tok/s':'tok/s ?';const ctx=w.last_prompt_tokens!=null?w.last_prompt_tokens+' prompt':'prompt ?';const up=w.uptime_s!=null?fmtUptime(w.uptime_s):'?';const mapping=s.llama.profile==='222'?(p===8081?'GPU 0,1':p===8082?'GPU 2,3':'GPU 4,5'):(p===8081?'GPU 0,1,2':'GPU 3,4,5');return '<div style="margin:8px 0;padding-top:6px;border-top:1px solid #333"><b>'+p+'</b> <span class="status '+(w.state||'down')+'">'+stateIcon(w)+' '+(w.status_text||'Down')+'</span> <span class="muted">• '+mapping+'</span><br><span class="muted">'+model+' • '+speed+' • '+ctx+' • up '+up+'</span><br><button onclick="workerAction(\'start\','+p+')">Start</button> <button onclick="workerAction(\'stop\','+p+')">Stop</button> <button onclick="workerAction(\'restart\','+p+')">Restart</button></div>'}).join('');
 gpus.innerHTML=s.gpu.devices.map(g=>`<div class="gpu"><b>#${g.index}</b><div><div>${g.name}</div><table><tr><td>Load</td><td>${g.utilization_pct??'?'}%</td><td>Temp</td><td class="${cls(g.temperature_c)}">${g.temperature_c??'?'}°C</td></tr><tr><td>Power</td><td>${g.power_w??'?'} W</td><td>Limit</td><td>${g.power_limit_w??'?'} W</td></tr><tr><td>VRAM</td><td>${mib(g.memory_used_mib)} / ${mib(g.memory_total_mib)}</td><td>Fan</td><td>${g.fan_pct==null?'N/A':g.fan_pct+'%'}</td></tr></table><div class="bar"><div class="fill" style="width:${Math.min(100,g.utilization_pct||0)}%"></div></div></div></div>`).join('');
 }catch(e){stamp.textContent='ERROR: '+e.message}
}
async function saveSample(){const v=parseFloat(v12.value);if(!Number.isFinite(v))return;saved.textContent='saving...';const r=await fetch('/api/psu-sample',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({voltage_12v:v,note:note.value})});const x=await r.json();saved.textContent=r.ok?'saved: '+x.sample.gpu_power_total_w+' W @ '+x.sample.voltage_12v+' V':'error: '+JSON.stringify(x)}
async function workerAction(action,port){workermsg.textContent=action+' '+port+'...';const r=await fetch('/api/workers/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,port})});const x=await r.json();workermsg.textContent=r.ok?'OK: '+action+' '+port:'ERROR: '+(x.detail||JSON.stringify(x));setTimeout(refresh,800)}
async function setProfile(p){workermsg.textContent='switching profile '+p+'...';const r=await fetch('/api/workers/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'profile',profile:p})});const x=await r.json();workermsg.textContent=r.ok?'OK: profile '+p:'ERROR: '+(x.detail||JSON.stringify(x));setTimeout(refresh,1200)}
function terminalUrl(){return location.protocol+'//'+location.hostname+':8091/'}
function openTerminal(){window.open(terminalUrl(),'_blank','noopener')}
function toggleTerminal(){const w=document.getElementById('termwrap'),f=document.getElementById('termframe');if(w.style.display==='none'){if(!f.src)f.src=terminalUrl();w.style.display='block'}else{w.style.display='none'}}
refresh();setInterval(refresh,2000);
</script></body></html>'''


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD

#!/usr/bin/env python3
import csv
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import psutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

APP_TITLE = "AI6 Host Monitor"
CSV_PATH = Path(os.environ.get("AI6_MONITOR_CSV", "/var/lib/ai6-monitor/psu-test.csv"))

app = FastAPI(title=APP_TITLE, version="1.0.0")
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
        gpus.append({
            "index": _int(cols[0]),
            "name": cols[1],
            "temperature_c": _float(cols[2]),
            "utilization_pct": _float(cols[3]),
            "memory_used_mib": _float(cols[4]),
            "memory_total_mib": _float(cols[5]),
            "power_w": _float(cols[6]),
            "power_limit_w": _float(cols[7]),
            "fan_pct": (lambda x: x if x is not None and 0 <= x <= 100 else None)(_float(cols[8])),
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
        with socket.create_connection(("127.0.0.1", port), timeout=0.4):
            return True
    except OSError:
        return False


def collect_stats():
    vm = psutil.virtual_memory()
    root = psutil.disk_usage("/")
    net = psutil.net_io_counters()
    gpus = gpu_stats()
    powers = [g["power_w"] for g in gpus if g["power_w"] is not None]
    temps = [g["temperature_c"] for g in gpus if g["temperature_c"] is not None]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": {
            "hostname": platform.node(),
            "kernel": platform.release(),
            "uptime_s": int(datetime.now().timestamp() - psutil.boot_time()),
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
            "devices": gpus,
        },
        "llama": {
            "8081": service_ok(8081),
            "8082": service_ok(8082),
        },
    }


class PsuSample(BaseModel):
    voltage_12v: float = Field(..., ge=0, le=20)
    note: str = Field(default="", max_length=200)


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


@app.get("/health")
def health():
    return {"status": "ok"}


DASHBOARD = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI6 Host Monitor</title>
<style>
body{font-family:system-ui,Arial,sans-serif;margin:20px;background:#111;color:#eee}h1{margin:0 0 6px}.muted{color:#aaa}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin-top:16px}.card{background:#1b1b1b;border:1px solid #333;border-radius:12px;padding:14px}.big{font-size:28px;font-weight:700}.gpu{display:grid;grid-template-columns:42px 1fr;gap:8px;border-top:1px solid #333;padding:9px 0}.gpu:first-child{border-top:0}.ok{color:#7be495}.warn{color:#ffd166}.bad{color:#ff6b6b}input,button{font:inherit;padding:8px;border-radius:8px;border:1px solid #555;background:#222;color:#eee}button{cursor:pointer}.bar{height:8px;background:#333;border-radius:5px;overflow:hidden;margin-top:5px}.fill{height:100%;background:#aaa;width:0%}table{width:100%;border-collapse:collapse}td{padding:4px 2px;border-bottom:1px solid #2d2d2d}</style>
</head><body>
<h1>AI6 Host Monitor</h1><div class="muted" id="stamp">loading...</div>
<div class="grid">
 <div class="card"><div>CPU</div><div class="big" id="cpu">-</div><div id="cpu2" class="muted"></div></div>
 <div class="card"><div>RAM</div><div class="big" id="ram">-</div><div id="ram2" class="muted"></div></div>
 <div class="card"><div>GPU total power</div><div class="big" id="power">-</div><div id="gputemp" class="muted"></div></div>
 <div class="card"><div>llama workers</div><div class="big" id="workers">-</div><div class="muted">8081 / 8082</div></div>
</div>
<div class="grid"><div class="card" style="grid-column:1/-1"><h3>GPUs</h3><div id="gpus"></div></div></div>
<div class="grid"><div class="card"><h3>PSU 12V sample</h3><p class="muted">Enter the multimeter reading. The current GPU power and temperatures will be logged to CSV on the host.</p><input id="v12" type="number" step="0.01" placeholder="12.05"><input id="note" placeholder="note, e.g. 400W"><button onclick="saveSample()">Save sample</button><div id="saved" class="muted"></div></div></div>
<div class="grid"><div class="card" style="grid-column:1/-1"><h3>Web Terminal</h3><p class="muted">Direct shell on the AI6 host. It runs as the normal Linux user and is protected by separate HTTP Basic authentication.</p><button onclick="openTerminal()">Open Web Terminal</button> <button onclick="toggleTerminal()">Show / hide below</button><div id="termwrap" style="display:none;margin-top:12px"><iframe id="termframe" title="AI6 Web Terminal" style="width:100%;height:520px;border:1px solid #333;border-radius:10px;background:#000"></iframe></div></div></div>
<script>
const mib=(v)=>v==null?'?':(v/1024).toFixed(2)+' GiB';
const cls=(t)=>t==null?'':(t>=80?'bad':t>=65?'warn':'ok');
async function refresh(){
 try{const r=await fetch('/api/stats');const s=await r.json();if(!r.ok)throw new Error(JSON.stringify(s));
 document.getElementById('stamp').textContent=s.host.hostname+' • '+new Date(s.timestamp).toLocaleString();
 cpu.textContent=s.cpu.usage_pct.toFixed(1)+'%'; cpu2.textContent='load '+s.cpu.load_1m.toFixed(2)+' / '+s.cpu.load_5m.toFixed(2)+' / '+s.cpu.load_15m.toFixed(2)+(s.cpu.temperature_c!=null?' • '+s.cpu.temperature_c.toFixed(0)+'°C':'');
 ram.textContent=s.memory.usage_pct.toFixed(1)+'%';ram2.textContent=(s.memory.used_bytes/1073741824).toFixed(2)+' / '+(s.memory.total_bytes/1073741824).toFixed(2)+' GiB';
 power.textContent=s.gpu.power_total_w.toFixed(1)+' W';gputemp.innerHTML='max temp <span class="'+cls(s.gpu.temperature_max_c)+'">'+(s.gpu.temperature_max_c??'?')+'°C</span>';
 workers.innerHTML=(s.llama['8081']?'✅':'❌')+' '+(s.llama['8082']?'✅':'❌');
 gpus.innerHTML=s.gpu.devices.map(g=>`<div class="gpu"><b>#${g.index}</b><div><div>${g.name}</div><table><tr><td>Load</td><td>${g.utilization_pct??'?'}%</td><td>Temp</td><td class="${cls(g.temperature_c)}">${g.temperature_c??'?'}°C</td></tr><tr><td>Power</td><td>${g.power_w??'?'} W</td><td>Limit</td><td>${g.power_limit_w??'?'} W</td></tr><tr><td>VRAM</td><td>${mib(g.memory_used_mib)} / ${mib(g.memory_total_mib)}</td><td>Fan</td><td>${g.fan_pct??'?'}%</td></tr></table><div class="bar"><div class="fill" style="width:${Math.min(100,g.utilization_pct||0)}%"></div></div></div></div>`).join('');
 }catch(e){stamp.textContent='ERROR: '+e.message}
}
async function saveSample(){const v=parseFloat(v12.value);if(!Number.isFinite(v))return;saved.textContent='saving...';const r=await fetch('/api/psu-sample',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({voltage_12v:v,note:note.value})});const x=await r.json();saved.textContent=r.ok?'saved: '+x.sample.gpu_power_total_w+' W @ '+x.sample.voltage_12v+' V':'error: '+JSON.stringify(x)}
function terminalUrl(){return location.protocol+'//'+location.hostname+':8091/'}
function openTerminal(){window.open(terminalUrl(),'_blank','noopener')}
function toggleTerminal(){const w=document.getElementById('termwrap'),f=document.getElementById('termframe');if(w.style.display==='none'){if(!f.src)f.src=terminalUrl();w.style.display='block'}else{w.style.display='none'}}
refresh();setInterval(refresh,2000);
</script></body></html>'''


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD

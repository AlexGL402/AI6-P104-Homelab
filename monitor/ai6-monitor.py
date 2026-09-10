#!/usr/bin/env python3
import csv
import io
import json
import os
import subprocess
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = os.getenv("AI6_MONITOR_HOST", "0.0.0.0")
PORT = int(os.getenv("AI6_MONITOR_PORT", "8090"))
START = time.time()


def read_meminfo():
    vals = {}
    with open('/proc/meminfo', 'r') as f:
        for line in f:
            k, v = line.split(':', 1)
            vals[k] = int(v.strip().split()[0]) * 1024
    total = vals.get('MemTotal', 0)
    avail = vals.get('MemAvailable', 0)
    used = max(0, total - avail)
    return {'total': total, 'used': used, 'available': avail,
            'percent': round(used * 100 / total, 1) if total else 0}


def cpu_sample(interval=0.12):
    def snap():
        p = open('/proc/stat').readline().split()[1:]
        x = list(map(int, p))
        idle = x[3] + (x[4] if len(x) > 4 else 0)
        return sum(x), idle
    t1, i1 = snap(); time.sleep(interval); t2, i2 = snap()
    dt = t2 - t1
    return round((1 - (i2-i1)/dt) * 100, 1) if dt else 0


def cpu_temp():
    temps = []
    base = '/sys/class/thermal'
    if os.path.isdir(base):
        for n in os.listdir(base):
            p = os.path.join(base, n, 'temp')
            try:
                v = float(open(p).read().strip())
                if v > 1000: v /= 1000
                if 0 < v < 130: temps.append(v)
            except Exception:
                pass
    return round(max(temps), 1) if temps else None


def gpu_stats():
    fields = ['index','name','utilization.gpu','temperature.gpu','power.draw','power.limit','memory.used','memory.total','fan.speed','clocks.current.graphics']
    cmd = ['nvidia-smi', '--query-gpu=' + ','.join(fields), '--format=csv,noheader,nounits']
    try:
        out = subprocess.check_output(cmd, text=True, timeout=4)
    except Exception as e:
        return [], str(e)
    gpus = []
    for line in out.strip().splitlines():
        r = next(csv.reader(io.StringIO(line), skipinitialspace=True))
        def num(s):
            try: return float(s)
            except: return None
        gpus.append({
            'index': int(r[0]), 'name': r[1], 'util': num(r[2]), 'temp': num(r[3]),
            'power': num(r[4]), 'power_limit': num(r[5]), 'mem_used': num(r[6]),
            'mem_total': num(r[7]), 'fan': num(r[8]), 'clock': num(r[9])
        })
    return gpus, None


def port_health(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=.6) as r:
            return r.status == 200
    except Exception:
        return False


def stats():
    gpus, err = gpu_stats()
    pwr = [g['power'] for g in gpus if g['power'] is not None]
    temps = [g['temp'] for g in gpus if g['temp'] is not None]
    load = os.getloadavg()
    return {
        'timestamp': int(time.time()),
        'uptime_s': int(time.time() - START),
        'cpu': {'percent': cpu_sample(), 'load1': round(load[0],2), 'load5': round(load[1],2), 'load15': round(load[2],2), 'temp': cpu_temp(), 'cores': os.cpu_count()},
        'memory': read_meminfo(),
        'gpus': gpus,
        'gpu_error': err,
        'gpu_count': len(gpus),
        'gpu_power_total': round(sum(pwr), 1),
        'gpu_temp_max': max(temps) if temps else None,
        'workers': {'8081': port_health(8081), '8082': port_health(8082)}
    }

HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AI6 Monitor</title>
<style>body{font-family:system-ui;background:#101216;color:#eee;margin:20px}.top,.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}.card{background:#1b1f26;border:1px solid #303642;border-radius:12px;padding:14px}.big{font-size:28px;font-weight:700}.ok{color:#72e58d}.bad{color:#ff7474}.bar{height:8px;background:#303642;border-radius:8px;overflow:hidden;margin-top:8px}.fill{height:100%;background:#aaa}.gpu{margin-top:12px}small{color:#aeb6c2}h1{margin-bottom:6px}</style></head><body>
<h1>AI6 Host Monitor</h1><small id="stamp">loading...</small><div class="top" id="summary"></div><h2>6× P104-100</h2><div class="grid" id="gpus"></div>
<script>function f(v,d=1){return v==null?'N/A':Number(v).toFixed(d)}function bar(v){return `<div class=bar><div class=fill style="width:${Math.max(0,Math.min(100,v||0))}%"></div></div>`}
async function go(){try{let d=await(await fetch('/api/stats')).json();stamp.textContent=new Date(d.timestamp*1000).toLocaleString();summary.innerHTML=`<div class=card><small>CPU</small><div class=big>${f(d.cpu.percent)}%</div>${bar(d.cpu.percent)}<div>Load ${d.cpu.load1} / ${d.cpu.load5}</div><div>Temp ${f(d.cpu.temp)} °C</div></div><div class=card><small>RAM</small><div class=big>${f(d.memory.percent)}%</div>${bar(d.memory.percent)}<div>${f(d.memory.used/1073741824,2)} / ${f(d.memory.total/1073741824,2)} GB</div></div><div class=card><small>GPU TOTAL POWER</small><div class=big>${f(d.gpu_power_total)} W</div><div>Max temp ${f(d.gpu_temp_max)} °C</div><div>GPUs ${d.gpu_count}</div></div><div class=card><small>LLAMA WORKERS</small><div>8081 <b class=${d.workers['8081']?'ok':'bad'}>${d.workers['8081']?'UP':'DOWN'}</b></div><div>8082 <b class=${d.workers['8082']?'ok':'bad'}>${d.workers['8082']?'UP':'DOWN'}</b></div></div>`;gpus.innerHTML=d.gpus.map(g=>`<div class="card gpu"><small>GPU ${g.index}</small><div><b>${g.name}</b></div><div class=big>${f(g.power)} W</div><div>Load ${f(g.util)}% ${bar(g.util)}</div><div>Temp ${f(g.temp)} °C</div><div>VRAM ${f(g.mem_used,0)} / ${f(g.mem_total,0)} MiB</div><div>Fan ${f(g.fan)}% · Clock ${f(g.clock,0)} MHz</div><div>Limit ${f(g.power_limit)} W</div></div>`).join('');}catch(e){stamp.textContent='ERROR: '+e} }go();setInterval(go,2000)</script></body></html>'''

class Handler(BaseHTTPRequestHandler):
    def send(self, code, ctype, data):
        b = data.encode(); self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == '/api/stats': self.send(200, 'application/json', json.dumps(stats()))
        elif self.path in ('/','/monitor'): self.send(200, 'text/html; charset=utf-8', HTML)
        elif self.path == '/health': self.send(200, 'application/json', '{"status":"ok"}')
        else: self.send(404, 'text/plain', 'not found')
    def log_message(self, fmt, *args): pass

if __name__ == '__main__':
    print(f'AI6 Monitor: http://{HOST}:{PORT}')
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()

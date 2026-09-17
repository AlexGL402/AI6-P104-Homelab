#!/usr/bin/env python3
"""Entry point for the auto-discovery AI6 monitor."""

import re
from pathlib import Path

import ai6_monitor_dynamic as dynamic

# Extend the host stats with the dedicated AI/model storage and per-socket CPU data.
_original_collect_stats = dynamic.collect_stats


def _cpu_socket_stats():
    per_cpu = dynamic.psutil.cpu_percent(interval=0.10, percpu=True)
    packages = {}

    for cpu_id, usage in enumerate(per_cpu):
        topo = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/topology")
        try:
            package_id = int((topo / "physical_package_id").read_text().strip())
        except (OSError, ValueError):
            package_id = 0
        try:
            core_id = int((topo / "core_id").read_text().strip())
        except (OSError, ValueError):
            core_id = cpu_id

        pkg = packages.setdefault(package_id, {"usage": [], "cpus": [], "cores": set()})
        pkg["usage"].append(float(usage))
        pkg["cpus"].append(cpu_id)
        pkg["cores"].add(core_id)

    package_temps = {}
    try:
        sensors = dynamic.psutil.sensors_temperatures(fahrenheit=False)
        for entries in sensors.values():
            for entry in entries:
                label = entry.label or ""
                match = re.search(r"Package\s+id\s+(\d+)", label, re.IGNORECASE)
                if match and entry.current is not None:
                    value = float(entry.current)
                    if -20 < value < 150:
                        package_temps[int(match.group(1))] = value
    except Exception:
        pass

    result = []
    for package_id in sorted(packages):
        pkg = packages[package_id]
        usage = sum(pkg["usage"]) / len(pkg["usage"]) if pkg["usage"] else 0.0
        result.append({
            "id": package_id,
            "usage_pct": round(usage, 1),
            "physical_cores": len(pkg["cores"]),
            "logical_cpus": len(pkg["cpus"]),
            "temperature_c": package_temps.get(package_id),
            "cpu_ids": pkg["cpus"],
        })
    return result


def collect_stats_with_host_details():
    stats = _original_collect_stats()

    try:
        ai = dynamic.psutil.disk_usage("/srv/ai")
        stats["ai_disk"] = {
            "mount": "/srv/ai",
            "used_bytes": ai.used,
            "total_bytes": ai.total,
            "free_bytes": ai.free,
            "usage_pct": ai.percent,
        }
    except (FileNotFoundError, OSError):
        stats["ai_disk"] = None

    stats.setdefault("cpu", {})["sockets"] = _cpu_socket_stats()
    return stats


dynamic.base.collect_stats = collect_stats_with_host_details

# ai6_monitor_dynamic builds JavaScript inside a Python raw string. Normalize
# the action-name escapes before serving the dashboard so the generated JS has
# exactly one backslash before the nested single quotes.
for action in ("start", "stop", "restart"):
    dynamic.base.DASHBOARD = dynamic.base.DASHBOARD.replace(
        "\\\\'" + action + "\\\\'",
        "\\'" + action + "\\'",
    )

# Show aggregate CPU plus one live card per physical CPU socket.
dynamic.base.DASHBOARD = dynamic.base.DASHBOARD.replace(
    '<div class="summary-card"><div class="label">CPU</div><div class="big" id="cpu">-</div><div id="cpu2" class="summary-sub"></div></div>',
    '<div class="summary-card"><div class="label">CPU total</div><div class="big" id="cpu">-</div><div id="cpu2" class="summary-sub"></div></div>\n'
    ' <div id="cpuSockets" style="display:contents"></div>',
)

dynamic.base.DASHBOARD = dynamic.base.DASHBOARD.replace(
    "cpu.textContent=s.cpu.usage_pct.toFixed(1)+'%';cpu2.textContent='load '+s.cpu.load_1m.toFixed(2)+' / '+s.cpu.load_5m.toFixed(2)+' / '+s.cpu.load_15m.toFixed(2)+(s.cpu.temperature_c!=null?' • '+s.cpu.temperature_c.toFixed(0)+'°C':'');",
    "cpu.textContent=s.cpu.usage_pct.toFixed(1)+'%';cpu2.textContent='load '+s.cpu.load_1m.toFixed(2)+' / '+s.cpu.load_5m.toFixed(2)+' / '+s.cpu.load_15m.toFixed(2)+(s.cpu.temperature_c!=null?' • max '+s.cpu.temperature_c.toFixed(0)+'°C':'');"
    "cpuSockets.innerHTML=(s.cpu.sockets||[]).map(c=>'<div class=\"summary-card\"><div class=\"label\">CPU '+c.id+'</div><div class=\"big\">'+c.usage_pct.toFixed(1)+'%</div><div class=\"summary-sub\">'+c.physical_cores+' cores / '+c.logical_cpus+' threads'+(c.temperature_c!=null?' • '+c.temperature_c.toFixed(0)+'°C':'')+'</div></div>').join('');",
)

# Show the system root and the dedicated AI NVMe as separate summary cards.
dynamic.base.DASHBOARD = dynamic.base.DASHBOARD.replace(
    '<div class="summary-card"><div class="label">Disk /</div><div class="big" id="disk">-</div><div id="disk2" class="summary-sub"></div></div>',
    '<div class="summary-card"><div class="label">Disk /</div><div class="big" id="disk">-</div><div id="disk2" class="summary-sub"></div></div>\n'
    ' <div class="summary-card"><div class="label">NVMe /srv/ai</div><div class="big" id="aidisk">-</div><div id="aidisk2" class="summary-sub"></div></div>',
)

dynamic.base.DASHBOARD = dynamic.base.DASHBOARD.replace(
    "disk.textContent=s.disk.usage_pct.toFixed(1)+'%';disk2.textContent=gib(s.disk.used_bytes)+' used • '+gib(s.disk.free_bytes)+' free / '+gib(s.disk.total_bytes);",
    "disk.textContent=s.disk.usage_pct.toFixed(1)+'%';disk2.textContent=gib(s.disk.used_bytes)+' used • '+gib(s.disk.free_bytes)+' free / '+gib(s.disk.total_bytes);"
    "if(s.ai_disk){aidisk.textContent=s.ai_disk.usage_pct.toFixed(1)+'%';aidisk2.textContent=gib(s.ai_disk.used_bytes)+' used • '+gib(s.ai_disk.free_bytes)+' free / '+gib(s.ai_disk.total_bytes);}else{aidisk.textContent='not mounted';aidisk2.textContent='/srv/ai unavailable';}",
)

app = dynamic.app

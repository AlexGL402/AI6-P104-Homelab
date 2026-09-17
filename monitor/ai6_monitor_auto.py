#!/usr/bin/env python3
"""Entry point for the auto-discovery AI6 monitor."""

import ai6_monitor_dynamic as dynamic

# Extend the host stats with the dedicated AI/model storage when it is mounted.
_original_collect_stats = dynamic.collect_stats


def collect_stats_with_ai_storage():
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
    return stats


dynamic.base.collect_stats = collect_stats_with_ai_storage

# ai6_monitor_dynamic builds JavaScript inside a Python raw string. Normalize
# the action-name escapes before serving the dashboard so the generated JS has
# exactly one backslash before the nested single quotes.
for action in ("start", "stop", "restart"):
    dynamic.base.DASHBOARD = dynamic.base.DASHBOARD.replace(
        "\\\\'" + action + "\\\\'",
        "\\'" + action + "\\'",
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

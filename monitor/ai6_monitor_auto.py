#!/usr/bin/env python3
"""Entry point for the auto-discovery AI6 monitor."""

import ai6_monitor_dynamic as dynamic
import ai6_vllm

# Add the dedicated vLLM tab/API on top of the dynamic monitor.
ai6_vllm.install()

# ai6_monitor_dynamic builds JavaScript inside a Python raw string. Normalize
# the action-name escapes before serving the dashboard so the generated JS has
# exactly one backslash before the nested single quotes.
for action in ("start", "stop", "restart"):
    dynamic.base.DASHBOARD = dynamic.base.DASHBOARD.replace(
        "\\\\'" + action + "\\\\'",
        "\\'" + action + "\\'",
    )

app = dynamic.app

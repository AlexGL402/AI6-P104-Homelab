#!/usr/bin/env python3
"""Entry point for the auto-discovery AI6 monitor."""

import ai6_monitor_dynamic as dynamic
import ai6_vllm
import ai6_miner
import ai6_insights

# Add the dedicated vLLM, miner and insights tabs/APIs on top of the dynamic monitor.
ai6_vllm.install()
ai6_miner.install()
ai6_insights.install()

# Finalize all top-level tabs in one place. This avoids nested-pane/layout
# corruption when optional modules are injected independently.
_tab_js = r"""
<script>
(function(){
  window.showTopTab = function(which){
    const tabs = {
      monitor: ['monitorTab','tabMonitorBtn'],
      vllm: ['vllmTab','tabVllmBtn'],
      benchs: ['benchsTab','tabBenchsBtn'],
      miner: ['minerTab','tabMinerBtn'],
      insights: ['insightsTab','tabInsightsBtn']
    };

    Object.entries(tabs).forEach(([name, ids]) => {
      const pane = document.getElementById(ids[0]);
      const btn = document.getElementById(ids[1]);
      if (pane) pane.style.display = (name === which) ? 'block' : 'none';
      if (btn) btn.classList.toggle('active', name === which);
    });

    if (which === 'vllm') {
      if (typeof loadVllmModels === 'function') loadVllmModels();
      if (typeof refreshGpuPowerLimitInfo === 'function') refreshGpuPowerLimitInfo();
      if (typeof refreshVllm === 'function') refreshVllm();
    } else if (which === 'benchs') {
      if (typeof loadBenchHistory === 'function') loadBenchHistory();
    } else if (which === 'miner') {
      if (typeof refreshMiner === 'function') refreshMiner();
      if (typeof refreshMinerLogs === 'function') refreshMinerLogs();
    } else if (which === 'insights') {
      if (typeof refreshInsights === 'function') refreshInsights();
    }
  };

  // Keep Monitor visible on first load even if a stale browser state exists.
  window.addEventListener('load', function(){
    const active = document.querySelector('.top-tabs .tab-btn.active');
    const which = active && active.id === 'tabVllmBtn' ? 'vllm'
      : active && active.id === 'tabBenchsBtn' ? 'benchs'
      : active && active.id === 'tabMinerBtn' ? 'miner'
      : active && active.id === 'tabInsightsBtn' ? 'insights'
      : 'monitor';
    window.showTopTab(which);
  });
})();
</script>
"""
if "</body>" not in dynamic.base.DASHBOARD:
    raise RuntimeError("Cannot finalize tabs: </body> marker not found")
dynamic.base.DASHBOARD = dynamic.base.DASHBOARD.replace("</body>", _tab_js + "\n</body>", 1)

# ai6_monitor_dynamic builds JavaScript inside a Python raw string. Normalize
# the action-name escapes before serving the dashboard so the generated JS has
# exactly one backslash before the nested single quotes.
for action in ("start", "stop", "restart"):
    dynamic.base.DASHBOARD = dynamic.base.DASHBOARD.replace(
        "\\\\'" + action + "\\\\'",
        "\\'" + action + "\\'",
    )

app = dynamic.app

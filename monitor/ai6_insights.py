#!/usr/bin/env python3
"""Insights / conclusions tab for AI6 benchmark history."""

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import ai6_monitor_dynamic as dynamic

base = dynamic.base

_STATE_DIR = Path(os.environ.get("AI6_MONITOR_STATE_DIR", str(Path.home() / ".local/state/ai6-monitor")))
_BENCH_STORE = Path(os.environ.get("AI6_VLLM_BENCH_STORE", str(_STATE_DIR / "vllm-benchmarks.json")))
_MARKET_HISTORY = Path(os.environ.get("AI6_MARKET_HISTORY", str(_STATE_DIR / "prl-market-history.json")))


def _rows():
    try:
        data = json.loads(_BENCH_STORE.read_text(encoding="utf-8"))
        return [x for x in data if isinstance(x, dict)]
    except Exception:
        return []


def _gpu_label(x):
    ds = x.get("gpu_devices") or []
    if ds:
        return " + ".join(f"GPU{g.get('index','?')} {g.get('name') or 'GPU'}" for g in ds)
    ids = x.get("gpu_ids") or [0]
    return " + ".join(f"GPU{i}" for i in ids)


def _eff(x):
    p = x.get("gpu_power_avg_w")
    return (float(x.get("aggregate_tok_s") or 0) / float(p)) if p else 0.0


def _confidence(n):
    if n >= 3:
        return "High"
    if n == 2:
        return "Medium"
    return "Low"


def _peer_key(x):
    return (
        x.get("model") or "",
        x.get("quantization") or "",
        _gpu_label(x),
        x.get("prompt_profile") or "",
        int(x.get("concurrency") or 0),
        int(x.get("max_tokens") or 0),
        bool(x.get("enable_thinking")),
    )


def _sweet_spot(rows):
    """Choose the lowest-power run within 3% of best throughput for a comparable workload."""
    groups = defaultdict(list)
    for x in rows:
        if x.get("gpu_power_limit_w") and x.get("aggregate_tok_s"):
            groups[_peer_key(x)].append(x)

    candidates = []
    for key, group in groups.items():
        pls = {round(float(x.get("gpu_power_limit_w") or 0)) for x in group}
        if len(pls) < 2:
            continue
        best_agg = max(float(x.get("aggregate_tok_s") or 0) for x in group)
        viable = [x for x in group if float(x.get("aggregate_tok_s") or 0) >= best_agg * 0.97]
        if not viable:
            continue
        chosen = max(viable, key=lambda x: (_eff(x), -float(x.get("gpu_power_limit_w") or 0)))
        penalty = 100 * (1 - float(chosen.get("aggregate_tok_s") or 0) / best_agg) if best_agg else 0
        candidates.append((len(group), _eff(chosen), chosen, best_agg, penalty))

    if not candidates:
        return None

    # Prefer well-sampled groups, then efficiency.
    candidates.sort(key=lambda z: (z[0], z[1]), reverse=True)
    n, eff, x, best_agg, penalty = candidates[0]
    return {
        "run": x,
        "samples": n,
        "efficiency": eff,
        "best_aggregate": best_agg,
        "throughput_penalty_pct": penalty,
        "confidence": _confidence(n),
    }


def _best_interactive(rows):
    valid = [
        x for x in rows
        if float(x.get("ttft_avg_s") or 999) <= 0.5
        and int(x.get("concurrency") or 0) <= 4
        and float(x.get("per_request_min_tok_s") or 0) > 0
    ]
    return max(valid, key=lambda x: float(x.get("per_request_min_tok_s") or 0), default=None)


def _best_batch(rows):
    return max(rows, key=lambda x: float(x.get("aggregate_tok_s") or 0), default=None)


def _best_long(rows):
    valid = [x for x in rows if int(x.get("max_tokens") or 0) >= 1024 and x.get("gpu_power_avg_w")]
    return max(valid, key=_eff, default=None)


def _anomalies(rows):
    notes = []
    for x in rows:
        if float(x.get("ttft_avg_s") or 0) >= 1.0 and str(x.get("prompt_profile") or "").lower() != "long":
            notes.append({
                "run_id": x.get("run_id"),
                "text": f"TTFT outlier {float(x.get('ttft_avg_s') or 0):.3f}s on {x.get('prompt_profile') or '-'} / conc {x.get('concurrency') or '-'} / PL {x.get('gpu_power_limit_w') or '-'}W",
            })
    return notes[-8:]



def _market_rows():
    try:
        data = json.loads(_MARKET_HISTORY.read_text(encoding="utf-8"))
        return [x for x in data if isinstance(x, dict) and x.get("ts")]
    except Exception:
        return []


def _pct_change(cur, old):
    try:
        cur = float(cur)
        old = float(old)
        return ((cur / old) - 1.0) * 100.0 if old else None
    except Exception:
        return None


def _value_before(rows, field, cutoff):
    candidates = [x for x in rows if int(x.get("ts") or 0) <= cutoff and x.get(field) is not None]
    if not candidates:
        return None
    return candidates[-1].get(field)


def _market_insights():
    rows = sorted(_market_rows(), key=lambda x: int(x.get("ts") or 0))
    current = rows[-1] if rows else {}

    # Pull a fresh current snapshot from the miner module when available. This
    # reuses the same cached market data that the Miner tab displays.
    try:
        import ai6_miner
        s = ai6_miner._status()
        p = s.get("profitability") or {}
        m = p.get("market") or {}
        current = {
            **current,
            "ts": int(time.time()),
            "price_usdt": m.get("prl_usdt"),
            "network_hashrate_hs": m.get("network_hashrate_hs"),
            "difficulty": m.get("difficulty"),
            "block_time_s": m.get("block_time_s"),
            "block_reward_prl": m.get("block_reward_prl"),
            "gross_kzt_day": p.get("gross_kzt_day"),
            "electricity_kzt_day": p.get("electricity_kzt_day"),
            "net_kzt_day": p.get("net_kzt_day"),
            "prl_day": p.get("prl_day"),
            "power_w": p.get("power_w"),
        }
    except Exception:
        pass

    now = int(time.time())
    changes = {}
    for label, seconds in (("24h", 86400), ("7d", 7 * 86400), ("30d", 30 * 86400)):
        changes[label] = {
            "price_pct": _pct_change(current.get("price_usdt"), _value_before(rows, "price_usdt", now - seconds)),
            "network_pct": _pct_change(current.get("network_hashrate_hs"), _value_before(rows, "network_hashrate_hs", now - seconds)),
            "net_pct": _pct_change(current.get("net_kzt_day"), _value_before(rows, "net_kzt_day", now - seconds)),
        }

    chart = [
        {
            "ts": int(x.get("ts") or 0),
            "price_usdt": x.get("price_usdt"),
            "network_ehs": (float(x.get("network_hashrate_hs")) / 1e18) if x.get("network_hashrate_hs") else None,
            "net_kzt_day": x.get("net_kzt_day"),
        }
        for x in rows[-4500:]
    ]

    return {
        "current": current,
        "changes": changes,
        "history": chart,
        "history_count": len(rows),
        "history_since": int(rows[0].get("ts") or 0) if rows else None,
    }


def _insights():
    rows = _rows()
    sweet = _sweet_spot(rows)
    interactive = _best_interactive(rows)
    batch = _best_batch(rows)
    long_run = _best_long(rows)

    gpu_names = sorted({_gpu_label(x) for x in rows if _gpu_label(x)})
    models = sorted({str(x.get("model") or "") for x in rows if x.get("model")})

    return {
        "count": len(rows),
        "gpus": gpu_names,
        "models": models,
        "sweet_spot": sweet,
        "interactive": interactive,
        "batch": batch,
        "long_decode": long_run,
        "market": _market_insights(),
        "anomalies": _anomalies(rows),
        "baseline": {
            "label": "PRE-MOD",
            "status": "captured",
            "control_points": [
                "Medium / conc 12 / 512 / 130W",
                "Medium / conc 12 / 1024 / 130W",
                "Medium / conc 12 / 2048 / 130W",
                "Medium / conc 12 / 2048 / 150W",
                "PearlHash / 130W",
            ],
        },
    }


@base.app.get("/api/insights")
def api_insights():
    return _insights()


def install():
    dashboard = base.DASHBOARD

    # Add Insights button after Miner.
    dashboard = dashboard.replace(
        '<button id="tabMinerBtn" class="tab-btn" onclick="showTopTab(\'miner\')">Miner</button>',
        '<button id="tabMinerBtn" class="tab-btn" onclick="showTopTab(\'miner\')">Miner</button>\n'
        '<button id="tabInsightsBtn" class="tab-btn" onclick="showTopTab(\'insights\')">Insights</button>',
        1,
    )

    html = r'''
<div id="insightsTab" style="display:none">
  <div class="section">
    <div class="section-head">
      <div>
        <div class="section-title">Insights / Conclusions</div>
        <div class="section-sub">Derived from saved benchmark history. Raw runs stay in Benchs; this tab tracks decisions and comparison baselines.</div>
      </div>
      <div id="insightsCount" class="muted">—</div>
    </div>

    <div class="insight-grid">
      <div class="insight-card accent">
        <span>Current sweet spot</span>
        <b id="insSweetValue">—</b>
        <small id="insSweetSub">Need comparable PL runs</small>
        <div id="insSweetConfidence" class="ins-confidence">—</div>
      </div>
      <div class="insight-card">
        <span>Interactive / coding</span>
        <b id="insInteractiveValue">—</b>
        <small id="insInteractiveSub">TTFT ≤ 0.5 s, conc ≤ 4</small>
      </div>
      <div class="insight-card">
        <span>Batch / API</span>
        <b id="insBatchValue">—</b>
        <small id="insBatchSub">Highest aggregate throughput</small>
      </div>
      <div class="insight-card">
        <span>Long decode</span>
        <b id="insLongValue">—</b>
        <small id="insLongSub">Best efficiency with output ≥ 1024</small>
      </div>
    </div>

    <div class="ins-panel ins-market-panel">
      <div class="ins-market-head">
        <div>
          <div class="ins-panel-title">PRL Market / Mining economics</div>
          <div class="ins-hint" id="prlMarketSince">Local history starts when this feature is enabled.</div>
        </div>
        <div class="ins-market-periods">
          <button class="prl-period active" data-period="24h" onclick="setPrlPeriod('24h',this)">24h</button>
          <button class="prl-period" data-period="7d" onclick="setPrlPeriod('7d',this)">7d</button>
          <button class="prl-period" data-period="30d" onclick="setPrlPeriod('30d',this)">30d</button>
        </div>
      </div>
      <div class="ins-market-grid">
        <div class="insight-card"><span>PRL price</span><b id="prlPrice">—</b><small id="prlPriceChange">collecting history</small></div>
        <div class="insight-card"><span>Network</span><b id="prlNetwork">—</b><small id="prlNetworkChange">collecting history</small></div>
        <div class="insight-card"><span>Difficulty</span><b id="prlDifficulty">—</b><small id="prlBlock">—</small></div>
        <div class="insight-card accent"><span>Our net/day</span><b id="prlNetDay">—</b><small id="prlNetChange">current hashrate + power</small></div>
      </div>
      <div class="prl-chart-wrap">
        <div class="prl-chart-title"><b id="prlChartTitle">PRL price • 24h</b><span id="prlChartRange">—</span></div>
        <svg id="prlChart" viewBox="0 0 1000 180" preserveAspectRatio="none" aria-label="PRL history chart"></svg>
      </div>
    </div>

    <div class="insight-two">
      <div class="ins-panel">
        <div class="ins-panel-title">PRE-MOD baseline</div>
        <div class="ins-baseline-head"><b>Captured</b><span>repeat these after capacitor modification</span></div>
        <div id="insBaseline" class="ins-list"></div>
      </div>
      <div class="ins-panel">
        <div class="ins-panel-title">GPU comparison</div>
        <div id="insGpuCompare" class="ins-list"><div class="muted">Loading…</div></div>
        <div class="ins-hint">CMP50 and P104 results will appear here automatically once comparable runs are saved.</div>
      </div>
    </div>

    <div class="ins-panel">
      <div class="ins-panel-title">Recommendations</div>
      <div id="insRecommendations" class="ins-rec-grid"></div>
    </div>

    <div class="ins-panel">
      <div class="ins-panel-title">Anomalies / notes</div>
      <div id="insAnomalies" class="ins-list"><div class="muted">No notes yet.</div></div>
    </div>
  </div>
</div>
'''
    if "</body>" not in dashboard:
        raise RuntimeError("Insights tab injection failed: </body> marker not found")
    if 'id="insightsTab"' not in dashboard:
        dashboard = dashboard.replace("</body>", html + "\n</body>", 1)

    css = r'''
.insight-grid{display:grid;grid-template-columns:repeat(4,minmax(180px,1fr));gap:9px;margin:10px 0}
.insight-card{background:#17191b;border:1px solid #30343a;border-radius:10px;padding:11px 12px;position:relative}.insight-card.accent{border-color:#3d7d4c}
.insight-card span,.insight-card small{display:block;color:#93999f;font-size:10px}.insight-card b{display:block;font-size:21px;color:#eee;margin:3px 0}.insight-card.accent b{color:#7be495}
.ins-confidence{display:inline-flex;margin-top:7px;border:1px solid #394047;border-radius:999px;padding:2px 7px;font-size:9px;color:#cfd3d7}
.ins-confidence.high{color:#7be495;border-color:#326a40;background:#142719}.ins-confidence.medium{color:#ffd56a;border-color:#6e5b21;background:#2b2513}.ins-confidence.low{color:#ff9a9a;border-color:#6f3030;background:#2b1717}
.insight-two{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-top:9px}.ins-panel{background:#151719;border:1px solid #30343a;border-radius:10px;padding:11px 12px;margin-top:9px}.ins-panel-title{font-weight:800;margin-bottom:8px}.ins-baseline-head{display:flex;gap:8px;align-items:center;margin-bottom:7px}.ins-baseline-head b{color:#79bfff}.ins-baseline-head span,.ins-hint{color:#8d949a;font-size:10px}
.ins-list{display:grid;gap:5px}.ins-list-row{display:flex;justify-content:space-between;gap:10px;padding:6px 8px;border:1px solid #292d31;border-radius:7px;background:#111315;font-size:11px}.ins-list-row b{color:#ddd}.ins-list-row span{color:#8f969c}
.ins-rec-grid{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));gap:7px}.ins-rec{border:1px solid #2b3034;border-radius:8px;padding:8px;background:#111315}.ins-rec b{display:block;color:#ddd;font-size:11px}.ins-rec span{display:block;color:#8f969c;font-size:10px;margin-top:3px}
.ins-market-head{display:flex;justify-content:space-between;gap:10px;align-items:center;flex-wrap:wrap}.ins-market-grid{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));gap:8px;margin-top:8px}.ins-market-periods{display:flex;gap:5px}.prl-period{background:#111315;color:#bbb;border:1px solid #343a40;border-radius:6px;padding:5px 9px;cursor:pointer}.prl-period.active{color:#7be495;border-color:#3d7d4c;background:#142719}.prl-chart-wrap{margin-top:9px;border:1px solid #292d31;border-radius:8px;background:#101214;padding:8px}.prl-chart-title{display:flex;justify-content:space-between;color:#8f969c;font-size:10px;margin-bottom:5px}.prl-chart-title b{color:#ddd}.prl-chart{width:100%;height:180px}.prl-gridline{stroke:#252a2e;stroke-width:1}.prl-line{fill:none;stroke:#7be495;stroke-width:2.2;vector-effect:non-scaling-stroke}.prl-empty{fill:#8f969c;font-size:18px}
@media(max-width:1100px){.insight-grid,.ins-rec-grid,.ins-market-grid{grid-template-columns:repeat(2,minmax(160px,1fr))}.insight-two{grid-template-columns:1fr}}
'''
    dashboard = dashboard.replace("</style>", css + "\n</style>", 1)

    # Wrap the existing Monitor/vLLM/Benchs/Miner switcher in JS instead of
    # rewriting its internals. This keeps the original tabs working even if
    # the Insights pane is missing or changed later.

    js = r'''
function insRunLabel(x){
 if(!x)return '—';
 return (x.prompt_profile||'—')+' • conc '+(x.concurrency??'—')+' • '+(x.max_tokens??'—')+' out • PL '+Math.round(x.gpu_power_limit_w||0)+'W';
}
function insRow(left,right){
 return '<div class="ins-list-row"><b>'+escHtml(left)+'</b><span>'+escHtml(right)+'</span></div>';
}

let prlPeriod='24h';
let prlMarketData=null;
function setPrlPeriod(p,btn){
 prlPeriod=p;
 document.querySelectorAll('.prl-period').forEach(x=>x.classList.toggle('active',x.dataset.period===p));
 if(prlMarketData)renderPrlMarket(prlMarketData);
}
function insPct(v){
 if(v==null||!Number.isFinite(Number(v)))return 'collecting history';
 const n=Number(v);
 return (n>=0?'+':'')+n.toFixed(1)+'% vs '+prlPeriod;
}
function insCompact(n){
 n=Number(n);
 if(!Number.isFinite(n))return '—';
 if(Math.abs(n)>=1e9)return (n/1e9).toFixed(2)+'B';
 if(Math.abs(n)>=1e6)return (n/1e6).toFixed(2)+'M';
 if(Math.abs(n)>=1e3)return (n/1e3).toFixed(1)+'k';
 return n.toFixed(0);
}
function renderPrlChart(history){
 const svg=document.getElementById('prlChart');
 if(!svg)return;
 const sec=prlPeriod==='24h'?86400:prlPeriod==='7d'?7*86400:30*86400;
 const now=Math.floor(Date.now()/1000);
 const pts=(history||[]).filter(x=>x.ts>=now-sec&&x.price_usdt!=null);
 if(pts.length<2){
  svg.innerHTML='<text x="500" y="95" text-anchor="middle" class="prl-empty">Collecting local history…</text>';
  prlChartRange.textContent=pts.length+' point'+(pts.length===1?'':'s');
  return;
 }
 const vals=pts.map(x=>Number(x.price_usdt)).filter(Number.isFinite);
 const min=Math.min(...vals),max=Math.max(...vals),span=(max-min)||1;
 const t0=pts[0].ts,t1=pts[pts.length-1].ts,dt=(t1-t0)||1;
 const xy=pts.map(x=>[(x.ts-t0)/dt*980+10,170-(Number(x.price_usdt)-min)/span*150]).filter(p=>Number.isFinite(p[1]));
 const d=xy.map((p,i)=>(i?'L':'M')+p[0].toFixed(1)+' '+p[1].toFixed(1)).join(' ');
 svg.innerHTML='<line x1="10" y1="20" x2="990" y2="20" class="prl-gridline"/><line x1="10" y1="95" x2="990" y2="95" class="prl-gridline"/><line x1="10" y1="170" x2="990" y2="170" class="prl-gridline"/><path d="'+d+'" class="prl-line"/>';
 prlChartRange.textContent='$'+min.toFixed(3)+' — $'+max.toFixed(3)+' • '+pts.length+' samples';
}
function renderPrlMarket(m){
 prlMarketData=m||{};
 const x=prlMarketData.current||{};
 const ch=(prlMarketData.changes||{})[prlPeriod]||{};
 prlPrice.textContent=x.price_usdt==null?'—':'$'+Number(x.price_usdt).toFixed(4);
 prlPriceChange.textContent=insPct(ch.price_pct);
 prlNetwork.textContent=x.network_hashrate_hs==null?'—':(Number(x.network_hashrate_hs)/1e18).toFixed(2)+' EH/s';
 prlNetworkChange.textContent=insPct(ch.network_pct);
 prlDifficulty.textContent=x.difficulty==null?'—':insCompact(x.difficulty);
 prlBlock.textContent=(x.block_time_s?'block ~'+Math.round(x.block_time_s)+'s':'')+(x.block_reward_prl?' • reward '+Number(x.block_reward_prl).toFixed(0)+' PRL':'');
 prlNetDay.textContent=x.net_kzt_day==null?'—':Math.round(Number(x.net_kzt_day)).toLocaleString('ru-RU')+' ₸';
 prlNetChange.textContent=insPct(ch.net_pct);
 prlChartTitle.textContent='PRL price • '+prlPeriod;
 if(prlMarketData.history_since){
  const dt=new Date(Number(prlMarketData.history_since)*1000);
  prlMarketSince.textContent='Local history: '+(prlMarketData.history_count||0)+' snapshots since '+dt.toLocaleString();
 }else{
  prlMarketSince.textContent='Local history starts now; 24h/7d/30d changes fill automatically.';
 }
 renderPrlChart(prlMarketData.history||[]);
}

async function refreshInsights(){
 try{
  const r=await fetch('/api/insights',{cache:'no-store'});
  const d=await r.json();
  if(!r.ok)throw new Error(d.detail||JSON.stringify(d));
  insightsCount.textContent=d.count+' saved runs';
  renderPrlMarket(d.market||{});

  const s=d.sweet_spot;
  if(s&&s.run){
   insSweetValue.textContent=Math.round(s.run.gpu_power_limit_w)+' W';
   insSweetSub.textContent=insRunLabel(s.run)+' • '+Number(s.run.aggregate_tok_s).toFixed(2)+' tok/s • '+Number(s.efficiency).toFixed(3)+' tok/s/W • '+Number(s.throughput_penalty_pct).toFixed(2)+'% from best';
   insSweetConfidence.textContent=s.confidence+' confidence • '+s.samples+' comparable runs';
   insSweetConfidence.className='ins-confidence '+String(s.confidence).toLowerCase();
  }else{
   insSweetValue.textContent='—';
   insSweetSub.textContent='Need comparable PL runs';
   insSweetConfidence.textContent='Low confidence';
   insSweetConfidence.className='ins-confidence low';
  }

  const i=d.interactive;
  insInteractiveValue.textContent=i?Number(i.per_request_min_tok_s).toFixed(2)+' tok/s':'—';
  insInteractiveSub.textContent=i?insRunLabel(i)+' • TTFT '+Number(i.ttft_avg_s).toFixed(3)+' s':'No qualifying run';

  const b=d.batch;
  insBatchValue.textContent=b?Number(b.aggregate_tok_s).toFixed(2)+' tok/s':'—';
  insBatchSub.textContent=b?insRunLabel(b):'No benchmark runs';

  const l=d.long_decode;
  insLongValue.textContent=l?Number(l.aggregate_tok_s/l.gpu_power_avg_w).toFixed(3)+' tok/s/W':'—';
  insLongSub.textContent=l?insRunLabel(l)+' • '+Number(l.aggregate_tok_s).toFixed(2)+' tok/s':'No 1024+ output run';

  insBaseline.innerHTML=(d.baseline.control_points||[]).map((x,n)=>insRow((n+1)+'. '+x,n<4?'vLLM control':'mining control')).join('');
  insGpuCompare.innerHTML=(d.gpus||[]).map(g=>insRow(g,'saved benchmark data')).join('')||'<div class="muted">No GPU data</div>';

  const sweetW=s&&s.run?Math.round(s.run.gpu_power_limit_w):'—';
  const recs=[
    ['Interactive coding',i?'Use '+insRunLabel(i):'Need more data'],
    ['Batch / API',b?'Best measured: '+Number(b.aggregate_tok_s).toFixed(2)+' tok/s':'Need more data'],
    ['Power efficiency',s?'Current sweet spot: '+sweetW+' W':'Need PL sweep'],
    ['AI idle','Use Miner tab; keep vLLM/miner mutually exclusive on the same GPU']
  ];
  insRecommendations.innerHTML=recs.map(x=>'<div class="ins-rec"><b>'+escHtml(x[0])+'</b><span>'+escHtml(x[1])+'</span></div>').join('');
  insAnomalies.innerHTML=(d.anomalies||[]).map(a=>insRow(a.run_id||'run',a.text||'')).join('')||'<div class="muted">No obvious TTFT outliers in saved runs.</div>';
 }catch(e){
  insightsCount.textContent='Insights error: '+e.message;
 }
}
'''
    dashboard = dashboard.replace("refresh();setInterval(refresh,2000);", js + "\nrefresh();setInterval(refresh,2000);", 1)

    base.DASHBOARD = dashboard
    dynamic.base.DASHBOARD = dashboard
    dynamic.app = base.app

"""
Credit Spread Attribution Dashboard — Data Fetcher v3
======================================================
Fallback chain:
  1. FRED (all 3 series)
  2. FRED OAS + yfinance ^TNX  (if only DGS10 fails)
  3. yfinance (all)
  4. Calibrated simulation

Output: C:\\Users\\xuhan\\Downloads\\credit_attribution_dashboard.html
"""

import json, os
from datetime import datetime
import numpy as np
import pandas as pd
import requests

# ── Configuration ──────────────────────────────────────────────────────────────
START_DATE  = "2025-01-01"
END_DATE    = datetime.today().strftime("%Y-%m-%d")
OUTPUT_FILE = r"C:\Users\xuhan\Downloads\credit_attribution_dashboard.html"

ETF_PARAMS = {
    "LQD": {"duration": 8.4,  "coupon": 4.8,  "convexity": 82.0},
    "HYG": {"duration": 4.1,  "coupon": 7.2,  "convexity": 22.0},
}

# ── FRED fetcher ───────────────────────────────────────────────────────────────
def fetch_fred_csv(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    print(f"  [FRED] {series_id} ...", end=" ", flush=True)
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        date_col = [c for c in df.columns if 'date' in c.lower()][0]
        val_col  = [c for c in df.columns if c != date_col][0]
        df = df.rename(columns={date_col: 'DATE', val_col: series_id})
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.set_index('DATE')
        df = df.replace('.', np.nan).astype(float).dropna()
        df = df.loc[START_DATE:END_DATE]
        print(f"OK ({len(df)} obs)")
        return df[series_id]
    except Exception as e:
        print(f"FAILED: {e}")
        return None

# ── yfinance fetcher ───────────────────────────────────────────────────────────
def fetch_yfinance_yield():
    """Fetch only 10y yield from yfinance ^TNX."""
    print("  [yfinance] ^TNX ...", end=" ", flush=True)
    try:
        import yfinance as yf
        df = yf.download("^TNX", start=START_DATE, end=END_DATE,
                         auto_adjust=True, progress=False)["Close"].squeeze().dropna()
        if df.empty or len(df) < 10:
            raise ValueError("No data")
        print(f"OK ({len(df)} days)")
        return df
    except Exception as e:
        print(f"FAILED: {e}")
        return None

def fetch_yfinance_all(ig_oas=None, hy_oas=None):
    """
    Fetch LQD, HYG, ^TNX from yfinance.
    If ig_oas/hy_oas already available from FRED, use those directly.
    Otherwise back-calculate OAS from price returns.
    """
    print("  [yfinance] LQD, HYG, ^TNX ...", end=" ", flush=True)
    try:
        import yfinance as yf
        raw = yf.download(["LQD", "HYG", "^TNX"], start=START_DATE, end=END_DATE,
                          auto_adjust=True, progress=False)["Close"].dropna()
        if raw.empty or len(raw) < 10:
            raise ValueError("No data returned")
        print(f"OK ({len(raw)} days)")

        rate_10y = raw["^TNX"]

        if ig_oas is not None and hy_oas is not None:
            # FRED OAS already available — just return with yfinance yield
            return {
                "lqd_oas":  ig_oas,
                "hyg_oas":  hy_oas,
                "rate_10y": rate_10y,
                "source":   "FRED OAS + yfinance ^TNX",
            }

        # Back-calculate OAS from price returns (only if FRED OAS unavailable)
        def backout_oas(price_series, duration, coupon, base_oas):
            daily_ret  = price_series.pct_change().dropna() * 100
            carry_d    = coupon / 252
            dr         = rate_10y.diff().dropna() / 100
            common     = daily_ret.index.intersection(dr.index)
            rate_c     = -duration * dr.loc[common] * 100
            dOAS_bp    = -(daily_ret.loc[common] - carry_d - rate_c) / duration * 10000
            # Smooth extreme daily moves (cap at ±50bp/day)
            dOAS_bp    = dOAS_bp.clip(-50, 50)
            oas        = base_oas + dOAS_bp.cumsum()
            return oas.clip(10, 2000)

        lqd_oas_yf = backout_oas(raw["LQD"], 8.4, 4.8, base_oas=100.0)
        hyg_oas_yf = backout_oas(raw["HYG"], 4.1, 7.2, base_oas=345.0)

        return {
            "lqd_oas":  lqd_oas_yf,
            "hyg_oas":  hyg_oas_yf,
            "rate_10y": rate_10y,
            "source":   "yfinance (LQD, HYG, ^TNX)",
        }
    except Exception as e:
        print(f"FAILED: {e}")
        return None

# ── Calibrated simulation ──────────────────────────────────────────────────────
def simulate_calibrated():
    print("  [simulation] Calibrated simulation (anchored to real levels).")
    dates = pd.date_range(START_DATE, END_DATE, freq="B")
    n = len(dates)
    rng = np.random.default_rng(42)

    def sim_oas(base, vol, drift):
        x = [base]
        for _ in range(n - 1):
            x.append(max(20, x[-1] + rng.normal(drift, vol)))
        return pd.Series(x, index=dates)

    def sim_rate(base, vol):
        x = [base]
        for _ in range(n - 1):
            x.append(max(1.0, x[-1] + rng.normal(-0.003, vol)))
        return pd.Series(x, index=dates)

    return {
        "lqd_oas":  sim_oas(108, 3.5, -0.06),
        "hyg_oas":  sim_oas(345, 9.0, -0.10),
        "rate_10y": sim_rate(4.42, 0.04),
        "source":   "calibrated simulation",
    }

# ── Attribution engine ─────────────────────────────────────────────────────────
def build_weekly(series):
    return series.resample("W-FRI").last().dropna()

def compute_attribution(oas_w, rate_w, duration, coupon, convexity):
    dOAS = oas_w.diff().dropna() / 10000
    dr   = rate_w.diff().dropna() / 100
    common = dOAS.index.intersection(dr.index)
    dOAS, dr = dOAS.loc[common], dr.loc[common]
    carry  = coupon / 52
    spread = -duration * dOAS * 100
    rate_  = -duration * dr   * 100
    convex =  0.5 * convexity * (dr + dOAS) ** 2 * 100
    return pd.DataFrame({
        "carry":     carry,
        "spread":    spread,
        "rate":      rate_,
        "convexity": convex,
        "total":     carry + spread + rate_ + convex,
    })

# ── HTML ───────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Credit Spread Attribution</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
:root{--bg:#0f1117;--bg2:#181c25;--bg3:#1e2330;--bdr:rgba(255,255,255,.08);
  --t:#e8eaf0;--t2:#8b90a0;--t3:#5a5f70;
  --ig:#185FA5;--hy:#993C1D;--rate:#534AB7;--spread:#1D9E75;--carry:#BA7517;--convex:#9E3D9E;
  --pos:#1D9E75;--neg:#C94040;--r:10px;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--t);padding:2rem;min-height:100vh}
h1{font-size:18px;font-weight:500;margin-bottom:4px}
.sub{font-size:13px;color:var(--t2);margin-bottom:1.5rem}
.badge{font-size:11px;padding:2px 8px;border-radius:20px;margin-left:8px;vertical-align:middle}
.badge-live{background:rgba(29,158,117,.15);color:var(--pos)}
.badge-yf{background:rgba(83,74,183,.15);color:#534AB7}
.badge-sim{background:rgba(186,117,23,.15);color:#BA7517}
.formula{background:var(--bg2);border:.5px solid var(--bdr);border-radius:var(--r);
  padding:10px 16px;font-size:12px;color:var(--t2);margin-bottom:1.25rem;line-height:2.1}
.formula code{font-family:"SF Mono","Fira Code",monospace;color:var(--t);font-size:12px}
.weight-row{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.25rem}
.wc{background:var(--bg2);border:.5px solid var(--bdr);border-radius:var(--r);padding:1rem 1.25rem}
.wc-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.wc-name{font-size:14px;font-weight:500}
.wc-tag{font-size:11px;padding:2px 8px;border-radius:20px}
.wc-pct{font-size:22px;font-weight:500;margin:6px 0 8px}
.slider{width:100%;margin-bottom:10px}
.wc-meta{font-size:12px;color:var(--t2);margin-bottom:10px}
.ab{margin:6px 0}
.ab-row{display:flex;justify-content:space-between;font-size:12px;color:var(--t2);margin-bottom:3px}
.ab-track{height:5px;background:var(--bg3);border-radius:3px;overflow:hidden}
.ab-fill{height:100%;border-radius:3px;transition:width .35s}
.metric-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(138px,1fr));gap:10px;margin-bottom:1.25rem}
.mc{background:var(--bg2);border:.5px solid var(--bdr);border-radius:var(--r);padding:13px 15px}
.mc-lbl{font-size:11px;color:var(--t2);margin-bottom:6px}
.mc-val{font-size:19px;font-weight:500}
.mc-sub{font-size:11px;margin-top:3px}
.pos{color:var(--pos)}.neg{color:var(--neg)}
.panel{background:var(--bg2);border:.5px solid var(--bdr);border-radius:var(--r);padding:1rem 1.25rem;margin-bottom:1rem}
.plbl{font-size:11px;font-weight:500;color:var(--t2);letter-spacing:.06em;text-transform:uppercase;margin-bottom:10px}
.tabs{display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap}
.tab{font-size:12px;padding:4px 12px;border-radius:var(--r);cursor:pointer;
  border:.5px solid rgba(255,255,255,.1);background:transparent;color:var(--t2)}
.tab.active{background:var(--bg3);color:var(--t);border-color:rgba(255,255,255,.2)}
.legend{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:10px;font-size:12px;color:var(--t2)}
.ld{width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:4px;vertical-align:middle}
#chartDiv{width:100%;height:280px}
table{width:100%;border-collapse:collapse;font-size:12px}
th{font-size:11px;font-weight:500;color:var(--t2);text-align:left;padding:5px 8px;border-bottom:.5px solid var(--bdr)}
td{padding:7px 8px;border-bottom:.5px solid var(--bdr)}
tr:last-child td{border-bottom:none;font-weight:500}
.mono{font-family:"SF Mono","Fira Code",monospace;font-size:11px;color:var(--t2)}
.note{font-size:11px;color:var(--t3);margin-top:.75rem}
</style>
</head>
<body>
<h1>Fixed income portfolio attribution <span class="badge __BADGE_CLASS__" id="dataBadge">__DATA_LABEL__</span></h1>
<p class="sub">LQD (IG) &nbsp;·&nbsp; HYG (HY) &nbsp;·&nbsp; __START_DATE__ – __END_DATE__ &nbsp;·&nbsp; Campisi framework with convexity</p>

<div class="formula">
  Portfolio return = w<sub>LQD</sub>&thinsp;·&thinsp;R<sub>LQD</sub> + w<sub>HYG</sub>&thinsp;·&thinsp;R<sub>HYG</sub>
  &nbsp;&nbsp;·&nbsp;&nbsp; R = <code>Rate</code> + <code>Spread</code> + <code>Carry</code> + <code>Convexity</code><br>
  <code>Spread = −Duration × ΔOAS</code> &nbsp;·&nbsp;
  <code>Rate = −Duration × Δr</code> &nbsp;·&nbsp;
  <code>Carry = coupon ÷ n</code> &nbsp;·&nbsp;
  <code>Convexity = ½ · C · (Δr + ΔOAS)²</code>
</div>

<div class="weight-row">
  <div class="wc">
    <div class="wc-header">
      <span class="wc-name" style="color:var(--ig)">LQD</span>
      <span class="wc-tag" style="background:rgba(24,95,165,.12);color:var(--ig)">Investment grade</span>
    </div>
    <div class="wc-pct" style="color:var(--ig)" id="lqdPct">60%</div>
    <input class="slider" type="range" min="0" max="100" value="60" id="lqdSlider" oninput="onSlider('lqd',this.value)">
    <div class="wc-meta">Dur: 8.4y &nbsp;·&nbsp; Coupon: ~4.8% &nbsp;·&nbsp; Convexity: 82</div>
    <div id="lqd-a-spread" class="ab"></div>
    <div id="lqd-a-carry"  class="ab"></div>
    <div id="lqd-a-rate"   class="ab"></div>
    <div id="lqd-a-convex" class="ab"></div>
  </div>
  <div class="wc">
    <div class="wc-header">
      <span class="wc-name" style="color:var(--hy)">HYG</span>
      <span class="wc-tag" style="background:rgba(153,60,29,.12);color:var(--hy)">High yield</span>
    </div>
    <div class="wc-pct" style="color:var(--hy)" id="hygPct">40%</div>
    <input class="slider" type="range" min="0" max="100" value="40" id="hygSlider" oninput="onSlider('hyg',this.value)">
    <div class="wc-meta">Dur: 4.1y &nbsp;·&nbsp; Coupon: ~7.2% &nbsp;·&nbsp; Convexity: 22</div>
    <div id="hyg-a-spread" class="ab"></div>
    <div id="hyg-a-carry"  class="ab"></div>
    <div id="hyg-a-rate"   class="ab"></div>
    <div id="hyg-a-convex" class="ab"></div>
  </div>
</div>

<div class="metric-row" id="metrics"></div>

<div class="panel">
  <p class="plbl">Portfolio return decomposition</p>
  <div class="tabs">
    <button class="tab active" onclick="switchTab('cum',this)">Cumulative attribution</button>
    <button class="tab" onclick="switchTab('weekly',this)">Weekly stacked</button>
    <button class="tab" onclick="switchTab('oas',this)">OAS levels</button>
    <button class="tab" onclick="switchTab('rate',this)">10y yield</button>
  </div>
  <div class="legend" id="legCum">
    <span><span class="ld" style="background:var(--rate)"></span>Rate</span>
    <span><span class="ld" style="background:var(--spread)"></span>Spread</span>
    <span><span class="ld" style="background:var(--carry)"></span>Carry</span>
    <span><span class="ld" style="background:var(--convex)"></span>Convexity</span>
    <span><span class="ld" style="background:#888"></span>Total</span>
  </div>
  <div class="legend" id="legOas" style="display:none">
    <span><span class="ld" style="background:var(--ig)"></span>LQD OAS (bp)</span>
    <span><span class="ld" style="background:var(--hy)"></span>HYG OAS (bp)</span>
  </div>
  <div class="legend" id="legRate" style="display:none">
    <span><span class="ld" style="background:var(--rate)"></span>US 10y yield (%)</span>
  </div>
  <div id="chartDiv"></div>
</div>

<div class="panel">
  <p class="plbl">Attribution summary</p>
  <table>
    <thead><tr>
      <th>Component</th><th>Formula</th>
      <th>LQD (weighted)</th><th>HYG (weighted)</th><th>Portfolio</th>
    </tr></thead>
    <tbody id="summaryTbody"></tbody>
  </table>
  <p class="note">Source: __DATA_SOURCE__ &nbsp;·&nbsp; Generated __GENERATED__</p>
</div>

<script>
const DATA = __DATA_JSON__;
let wLQD = 0.60, wHYG = 0.40, currentTab = 'cum';

function sum(arr, k) { return arr.reduce((a, b) => a + b[k], 0); }

function getTotals() {
  const ls=sum(DATA.lqd,'spread'), lc=sum(DATA.lqd,'carry'), lr=sum(DATA.lqd,'rate'), lx=sum(DATA.lqd,'convexity');
  const hs=sum(DATA.hyg,'spread'), hc=sum(DATA.hyg,'carry'), hr=sum(DATA.hyg,'rate'), hx=sum(DATA.hyg,'convexity');
  return {
    wlqd: { spread:wLQD*ls, carry:wLQD*lc, rate:wLQD*lr, convexity:wLQD*lx, total:wLQD*(ls+lc+lr+lx) },
    whyg: { spread:wHYG*hs, carry:wHYG*hc, rate:wHYG*hr, convexity:wHYG*hx, total:wHYG*(hs+hc+hr+hx) },
    port: { spread:wLQD*ls+wHYG*hs, carry:wLQD*lc+wHYG*hc, rate:wLQD*lr+wHYG*hr, convexity:wLQD*lx+wHYG*hx,
            total:wLQD*(ls+lc+lr+lx)+wHYG*(hs+hc+hr+hx) },
  };
}

function makeBar(id, label, val, color) {
  const pct = Math.min(100, Math.abs(val) / 3 * 100);
  const vc  = val >= 0 ? 'var(--pos)' : 'var(--neg)';
  document.getElementById(id).innerHTML = `
    <div class="ab-row"><span>${label}</span>
      <span style="color:${vc}">${val>=0?'+':''}${val.toFixed(2)}%</span></div>
    <div class="ab-track">
      <div class="ab-fill" style="width:${pct}%;background:${color}"></div></div>`;
}

function updateCards(t) {
  document.getElementById('lqdPct').textContent = Math.round(wLQD*100) + '%';
  document.getElementById('hygPct').textContent = Math.round(wHYG*100) + '%';
  makeBar('lqd-a-spread','Spread',    t.wlqd.spread,    'var(--spread)');
  makeBar('lqd-a-carry', 'Carry',     t.wlqd.carry,     'var(--carry)');
  makeBar('lqd-a-rate',  'Rate',      t.wlqd.rate,      'var(--rate)');
  makeBar('lqd-a-convex','Convexity', t.wlqd.convexity, 'var(--convex)');
  makeBar('hyg-a-spread','Spread',    t.whyg.spread,    'var(--spread)');
  makeBar('hyg-a-carry', 'Carry',     t.whyg.carry,     'var(--carry)');
  makeBar('hyg-a-rate',  'Rate',      t.whyg.rate,      'var(--rate)');
  makeBar('hyg-a-convex','Convexity', t.whyg.convexity, 'var(--convex)');
}

function updateMetrics(t) {
  const N = DATA.lqd_oas.length;
  const oasNow   = wLQD*DATA.lqd_oas[N-1] + wHYG*DATA.hyg_oas[N-1];
  const oasStart = wLQD*DATA.lqd_oas[0]   + wHYG*DATA.hyg_oas[0];
  const oasChg   = oasNow - oasStart;
  const r10Now   = DATA.rate_10y[N-1];
  const items = [
    { l:'Portfolio total',    v:(t.port.total>=0?'+':'')+t.port.total.toFixed(2)+'%',       s:'Rate+Spread+Carry+Convexity', p:t.port.total>=0 },
    { l:'Wtd avg OAS (now)',  v:oasNow.toFixed(0)+' bp',
      s:(oasChg>=0?'+':'')+oasChg.toFixed(0)+' bp vs start',                               p:oasChg<0 },
    { l:'Spread contrib',     v:(t.port.spread>=0?'+':'')+t.port.spread.toFixed(2)+'%',     s:'−Dur × ΔOAS',      p:t.port.spread>=0 },
    { l:'Rate contrib',       v:(t.port.rate>=0?'+':'')+t.port.rate.toFixed(2)+'%',         s:'−Dur × Δ10y',      p:t.port.rate>=0 },
    { l:'Carry',              v:'+'+t.port.carry.toFixed(2)+'%',                            s:'Coupon income',    p:true },
    { l:'Convexity contrib',  v:(t.port.convexity>=0?'+':'')+t.port.convexity.toFixed(2)+'%', s:'½·C·(Δr+ΔOAS)²', p:t.port.convexity>=0 },
    { l:'10y yield (latest)', v:r10Now.toFixed(2)+'%',                                      s:DATA.source,        p:r10Now<4.5 },
  ];
  document.getElementById('metrics').innerHTML = items.map(m => `
    <div class="mc">
      <div class="mc-lbl">${m.l}</div>
      <div class="mc-val ${m.p?'pos':'neg'}">${m.v}</div>
      <div class="mc-sub ${m.p?'pos':'neg'}">${m.s}</div>
    </div>`).join('');
}

function updateTable(t) {
  const rows = [
    { c:'Rate contribution',   f:'−Dur × Δ10y',     lqd:t.wlqd.rate,      hyg:t.whyg.rate,      p:t.port.rate },
    { c:'Spread contribution', f:'−Dur × ΔOAS',      lqd:t.wlqd.spread,    hyg:t.whyg.spread,    p:t.port.spread },
    { c:'Carry',               f:'coupon ÷ n',        lqd:t.wlqd.carry,     hyg:t.whyg.carry,     p:t.port.carry },
    { c:'Convexity',           f:'½·C·(Δr+ΔOAS)²',  lqd:t.wlqd.convexity, hyg:t.whyg.convexity, p:t.port.convexity },
    { c:'Total return',        f:'Sum',               lqd:t.wlqd.total,     hyg:t.whyg.total,     p:t.port.total },
  ];
  const sep = 'border-top:.5px solid rgba(255,255,255,.08);';
  document.getElementById('summaryTbody').innerHTML = rows.map((r, i) => {
    const s = i === 4 ? sep : '';
    const c = v => v >= 0 ? 'var(--pos)' : 'var(--neg)';
    const f = v => (v >= 0 ? '+' : '') + v.toFixed(2) + '%';
    return `<tr>
      <td style="${s}">${r.c}</td>
      <td class="mono" style="${s}">${r.f}</td>
      <td style="color:${c(r.lqd)};${s}">${f(r.lqd)}</td>
      <td style="color:${c(r.hyg)};${s}">${f(r.hyg)}</td>
      <td style="color:${c(r.p)};${s}">${f(r.p)}</td>
    </tr>`;
  }).join('');
}

const C = { rate:'#534AB7', spread:'#1D9E75', carry:'#BA7517', convex:'#9E3D9E', total:'#888', ig:'#185FA5', hy:'#993C1D' };
const LY = {
  paper_bgcolor:'transparent', plot_bgcolor:'transparent',
  margin:{ l:44, r:20, t:10, b:40 },
  font:{ family:'-apple-system,sans-serif', size:11, color:'#8b90a0' },
  xaxis:{ showgrid:false, tickcolor:'transparent', linecolor:'rgba(255,255,255,.08)' },
  yaxis:{ gridcolor:'rgba(255,255,255,.05)', tickcolor:'transparent', linecolor:'rgba(255,255,255,.08)' },
  legend:{ bgcolor:'transparent' },
  hovermode:'x unified',
};

function mkLine(name, x, y, color, dash) {
  return { x, y, name, mode:'lines', line:{ color, width:1.5, dash: dash||'solid' },
           hovertemplate:'%{y:.2f}%<extra>' + name + '</extra>' };
}
function mkBar(name, x, y, color) {
  return { x, y, name, type:'bar', marker:{ color },
           hovertemplate:'%{y:.3f}%<extra>' + name + '</extra>' };
}

function buildCumChart() {
  let cr=0, cs=0, cc=0, cx=0, ct=0;
  const R=[], S=[], C2=[], X=[], T=[];
  DATA.lqd.forEach((_, i) => {
    cr += wLQD*DATA.lqd[i].rate      + wHYG*DATA.hyg[i].rate;
    cs += wLQD*DATA.lqd[i].spread    + wHYG*DATA.hyg[i].spread;
    cc += wLQD*DATA.lqd[i].carry     + wHYG*DATA.hyg[i].carry;
    cx += wLQD*DATA.lqd[i].convexity + wHYG*DATA.hyg[i].convexity;
    ct += wLQD*DATA.lqd[i].total     + wHYG*DATA.hyg[i].total;
    R.push(+cr.toFixed(3)); S.push(+cs.toFixed(3));
    C2.push(+cc.toFixed(3)); X.push(+cx.toFixed(3)); T.push(+ct.toFixed(3));
  });
  Plotly.newPlot('chartDiv', [
    mkLine('Rate',      DATA.labels, R,  C.rate),
    mkLine('Spread',    DATA.labels, S,  C.spread),
    mkLine('Carry',     DATA.labels, C2, C.carry),
    mkLine('Convexity', DATA.labels, X,  C.convex),
    mkLine('Total',     DATA.labels, T,  C.total, 'dash'),
  ], { ...LY, yaxis:{ ...LY.yaxis, ticksuffix:'%' } }, { responsive:true, displayModeBar:false });
}

function buildWeeklyChart() {
  const pR=[], pS=[], pC=[], pX=[];
  DATA.lqd.forEach((_, i) => {
    pR.push(+(wLQD*DATA.lqd[i].rate      + wHYG*DATA.hyg[i].rate     ).toFixed(4));
    pS.push(+(wLQD*DATA.lqd[i].spread    + wHYG*DATA.hyg[i].spread   ).toFixed(4));
    pC.push(+(wLQD*DATA.lqd[i].carry     + wHYG*DATA.hyg[i].carry    ).toFixed(4));
    pX.push(+(wLQD*DATA.lqd[i].convexity + wHYG*DATA.hyg[i].convexity).toFixed(4));
  });
  Plotly.newPlot('chartDiv', [
    mkBar('Rate',      DATA.labels, pR, 'rgba(83,74,183,.75)'),
    mkBar('Spread',    DATA.labels, pS, 'rgba(29,158,117,.75)'),
    mkBar('Carry',     DATA.labels, pC, 'rgba(186,117,23,.75)'),
    mkBar('Convexity', DATA.labels, pX, 'rgba(158,61,158,.75)'),
  ], { ...LY, barmode:'stack', yaxis:{ ...LY.yaxis, ticksuffix:'%' } }, { responsive:true, displayModeBar:false });
}

function buildOASChart() {
  Plotly.newPlot('chartDiv', [
    { x:DATA.oas_dates, y:DATA.lqd_oas, name:'LQD OAS', mode:'lines',
      line:{ color:C.ig, width:1.5 }, hovertemplate:'%{y:.0f} bp<extra>LQD</extra>' },
    { x:DATA.oas_dates, y:DATA.hyg_oas, name:'HYG OAS', mode:'lines',
      line:{ color:C.hy, width:1.5 }, yaxis:'y2', hovertemplate:'%{y:.0f} bp<extra>HYG</extra>' },
  ], {
    ...LY,
    yaxis:  { ...LY.yaxis, title:{ text:'LQD OAS (bp)', font:{ color:C.ig, size:11 } } },
    yaxis2: { overlaying:'y', side:'right', gridcolor:'transparent', tickcolor:'transparent',
               linecolor:'rgba(255,255,255,.08)', title:{ text:'HYG OAS (bp)', font:{ color:C.hy, size:11 } } },
  }, { responsive:true, displayModeBar:false });
}

function buildRateChart() {
  Plotly.newPlot('chartDiv', [
    { x:DATA.oas_dates, y:DATA.rate_10y, name:'10y yield', mode:'lines',
      line:{ color:C.rate, width:1.5 }, hovertemplate:'%{y:.2f}%<extra>10y yield</extra>' },
  ], { ...LY, yaxis:{ ...LY.yaxis, ticksuffix:'%' } }, { responsive:true, displayModeBar:false });
}

function redraw() {
  const t = getTotals();
  updateCards(t); updateMetrics(t); updateTable(t);
  if      (currentTab === 'cum')    buildCumChart();
  else if (currentTab === 'weekly') buildWeeklyChart();
  else if (currentTab === 'oas')    buildOASChart();
  else                              buildRateChart();
}

function onSlider(which, val) {
  const v = parseInt(val) / 100;
  if (which === 'lqd') { wLQD = v; wHYG = Math.max(0, 1-v); document.getElementById('hygSlider').value = Math.round(wHYG*100); }
  else                 { wHYG = v; wLQD = Math.max(0, 1-v); document.getElementById('lqdSlider').value = Math.round(wLQD*100); }
  redraw();
}

function switchTab(tab, btn) {
  currentTab = tab;
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('legCum').style.display  = (tab==='cum'||tab==='weekly') ? 'flex' : 'none';
  document.getElementById('legOas').style.display  = tab === 'oas'  ? 'flex' : 'none';
  document.getElementById('legRate').style.display = tab === 'rate' ? 'flex' : 'none';
  redraw();
}

redraw();
</script>
</body>
</html>"""

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Credit Attribution Dashboard — Data Fetcher v3")
    print("=" * 60)
    print(f"\n[1/3] Fetching market data ({START_DATE} → {END_DATE})...")

    ig_oas = fetch_fred_csv("BAMLC0A0CM")
    if ig_oas is not None:
      ig_oas = ig_oas * 100  # % → bp

    hy_oas = fetch_fred_csv("BAMLH0A0HYM2")
    if hy_oas is not None:
      hy_oas = hy_oas * 100  # % → bp
    
    r10    = fetch_fred_csv("DGS10")

    # Determine data source with smart fallback
    if all(x is not None for x in [ig_oas, hy_oas, r10]):
        df = pd.DataFrame({"ig": ig_oas, "hy": hy_oas, "r10": r10})
        df = df.loc[START_DATE:END_DATE].dropna()
        market = { "lqd_oas": df["ig"], "hyg_oas": df["hy"], "rate_10y": df["r10"],
                   "source": "FRED (BAMLC0A0CM · BAMLH0A0HYM2 · DGS10)" }
        badge_class, data_label = "badge-live", "live · FRED"
        data_source = "FRED — ICE BofA OAS indices + DGS10"

    elif ig_oas is not None and hy_oas is not None and r10 is None:
        # OAS OK but DGS10 failed — try yfinance yield only
        print("\n  FRED OAS OK but DGS10 failed — fetching yield from yfinance...")
        tnx = fetch_yfinance_yield()
        if tnx is not None:
            market = { "lqd_oas": ig_oas, "hyg_oas": hy_oas, "rate_10y": tnx,
                       "source": "FRED OAS + yfinance ^TNX" }
            badge_class, data_label = "badge-yf", "live · FRED + yf"
            data_source = "FRED (OAS) + yfinance (^TNX)"
        else:
            print("  yfinance yield also failed — falling back to full yfinance...")
            market = fetch_yfinance_all(ig_oas, hy_oas)
            if market is None:
                market = simulate_calibrated()
                badge_class, data_label = "badge-sim", "calibrated simulation"
                data_source = "Calibrated simulation"
            else:
                badge_class, data_label = "badge-yf", "live · yfinance"
                data_source = "FRED OAS + yfinance"

    else:
        # FRED mostly failed — try full yfinance
        print("\n  FRED incomplete — trying yfinance...")
        market = fetch_yfinance_all()
        if market is not None:
            badge_class, data_label = "badge-yf", "live · yfinance"
            data_source = "yfinance (LQD, HYG, ^TNX)"
        else:
            market = simulate_calibrated()
            badge_class, data_label = "badge-sim", "calibrated simulation"
            data_source = "Calibrated simulation (anchored to real levels)"

    print("\n[2/3] Computing weekly attribution...")
    lqd_oas_w  = build_weekly(market["lqd_oas"])
    hyg_oas_w  = build_weekly(market["hyg_oas"])
    rate_10y_w = build_weekly(market["rate_10y"])

    common_w = lqd_oas_w.index.intersection(hyg_oas_w.index).intersection(rate_10y_w.index)
    lqd_oas_w  = lqd_oas_w.loc[common_w]
    hyg_oas_w  = hyg_oas_w.loc[common_w]
    rate_10y_w = rate_10y_w.loc[common_w]

    print(f"  LQD OAS range: {lqd_oas_w.min():.1f} – {lqd_oas_w.max():.1f} bp")
    print(f"  HYG OAS range: {hyg_oas_w.min():.1f} – {hyg_oas_w.max():.1f} bp")
    print(f"  LQD OAS first: {lqd_oas_w.iloc[0]:.1f}, last: {lqd_oas_w.iloc[-1]:.1f}")
    print(f"  HYG OAS first: {hyg_oas_w.iloc[0]:.1f}, last: {hyg_oas_w.iloc[-1]:.1f}")
    print(f"  10y yield range: {rate_10y_w.min():.2f} – {rate_10y_w.max():.2f} %")
    print(f"  10y yield first: {rate_10y_w.iloc[0]:.2f}, last: {rate_10y_w.iloc[-1]:.2f}")

    lqd_attr = compute_attribution(lqd_oas_w, rate_10y_w, **ETF_PARAMS["LQD"])
    hyg_attr = compute_attribution(hyg_oas_w, rate_10y_w, **ETF_PARAMS["HYG"])

    common_a = lqd_attr.index.intersection(hyg_attr.index)
    lqd_attr = lqd_attr.loc[common_a]
    hyg_attr = hyg_attr.loc[common_a]

    print(f"  {len(lqd_attr)} weekly periods")
    print(f"  LQD total return: {lqd_attr['total'].sum():.2f}%")
    print(f"  HYG total return: {hyg_attr['total'].sum():.2f}%")

    print("\n[3/3] Building HTML...")

    chart_labels = [d.strftime("%b %d") for d in lqd_attr.index]
    oas_dates    = [d.strftime("%Y-%m-%d") for d in lqd_oas_w.loc[common_a].index]

    def to_list(df):
        return [{ "spread":    round(float(r.spread),    4),
                  "carry":     round(float(r.carry),     4),
                  "rate":      round(float(r.rate),      4),
                  "convexity": round(float(r.convexity), 4),
                  "total":     round(float(r.total),     4) }
                for _, r in df.iterrows()]

    data_json = json.dumps({
        "labels":    chart_labels,
        "oas_dates": oas_dates,
        "lqd_oas":   [round(float(x), 2) for x in lqd_oas_w.loc[common_a]],
        "hyg_oas":   [round(float(x), 2) for x in hyg_oas_w.loc[common_a]],
        "rate_10y":  [round(float(x), 3) for x in rate_10y_w.loc[common_a]],
        "lqd":       to_list(lqd_attr),
        "hyg":       to_list(hyg_attr),
        "source":    market["source"],
    })

    html = HTML
    html = html.replace("__BADGE_CLASS__", badge_class)
    html = html.replace("__DATA_LABEL__",  data_label)
    html = html.replace("__START_DATE__",  START_DATE)
    html = html.replace("__END_DATE__",    END_DATE)
    html = html.replace("__DATA_SOURCE__", data_source)
    html = html.replace("__GENERATED__",   datetime.today().strftime("%Y-%m-%d"))
    html = html.replace("__DATA_JSON__",   data_json)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_FILE)), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✓  Saved → {OUTPUT_FILE}")
    print("   Open in any browser — no server needed.\n")

if __name__ == "__main__":
    main()

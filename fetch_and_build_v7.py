"""
Fixed Income Portfolio Attribution Dashboard — v7
==================================================
Key Rate Duration (KRD) implementation:
  LQD (dur=8.4y): DGS5 + DGS30
  HYG (dur=4.1y): DGS2 + DGS5

New in v7:
  - Yields now come from the US Treasury par yield curve (real 2/5/10/30,
    no API key) — FRED's DGS series time out on some networks. OAS still
    from FRED. Fallback chain: Treasury → yfinance proxy → simulation.

v6 added the yield-curve PCA module:
  - DGS/UST 2/5/10/30 weekly changes → covariance → eigendecomposition →
    Level / Slope / Curvature, with the rate component split into orthogonal
    factor contributions that reconcile with the KRD rate total.

Fallback chain: FRED → yfinance → calibrated simulation (now correlated curve)
Output: C:\\Users\\xuhan\\Downloads\\credit_attribution_dashboard.html
"""

import json, os
from datetime import datetime
import numpy as np
import pandas as pd

START_DATE  = "2025-01-01"
END_DATE    = datetime.today().strftime("%Y-%m-%d")
OUTPUT_FILE = os.environ.get("FIA_OUTPUT",
                             r"C:\Users\xuhan\Downloads\credit_attribution_dashboard.html")

# ── KRD parameters ─────────────────────────────────────────────────────────────
# Linear interpolation: KRD_near = dur * (far_node - dur) / (far_node - near_node)
#                       KRD_far  = dur * (dur - near_node) / (far_node - near_node)
def calc_krd(duration, near_node, far_node):
    span = far_node - near_node
    krd_near = duration * (far_node - duration) / span
    krd_far  = duration * (duration - near_node) / span
    # Clip to avoid negative KRD when duration is outside the node range
    krd_near = max(0.0, krd_near)
    krd_far  = max(0.0, krd_far)
    return krd_near, krd_far

LQD_DUR, HYG_DUR = 8.4, 4.1
LQD_KRD_5, LQD_KRD_30 = calc_krd(LQD_DUR, 5, 30)   # nodes: DGS5, DGS30
HYG_KRD_2, HYG_KRD_5  = calc_krd(HYG_DUR, 2, 5)    # nodes: DGS2, DGS5

ETF_PARAMS = {
    "LQD": {
        "duration": LQD_DUR, "coupon": 4.8, "convexity": 82.0,
        "krd_near": LQD_KRD_5,  "krd_far": LQD_KRD_30,
        "near_key": "r5",        "far_key": "r30",
        "near_lbl": "5y",        "far_lbl": "30y",
    },
    "HYG": {
        "duration": HYG_DUR, "coupon": 7.2, "convexity": 22.0,
        "krd_near": HYG_KRD_2,  "krd_far": HYG_KRD_5,
        "near_key": "r2",        "far_key": "r5",
        "near_lbl": "2y",        "far_lbl": "5y",
    },
}

print(f"KRD — LQD:  5y={LQD_KRD_5:.2f}  30y={LQD_KRD_30:.2f}  (sum={LQD_KRD_5+LQD_KRD_30:.2f}, dur={LQD_DUR})")
print(f"KRD — HYG:  2y={HYG_KRD_2:.2f}   5y={HYG_KRD_5:.2f}   (sum={HYG_KRD_2+HYG_KRD_5:.2f}, dur={HYG_DUR})")

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
        df = df.set_index('DATE').replace('.', np.nan).astype(float).dropna()
        df = df.loc[START_DATE:END_DATE]
        print(f"OK ({len(df)} obs)")
        return df[series_id]
    except Exception as e:
        print(f"FAILED: {e}")
        return None

def fetch_yfinance_yields():
    """
    Fetch yield curve proxies from yfinance:
      ^IRX = 13-week T-bill  → proxy for 2y
      ^FVX = 5-year yield    → proxy for 5y
      ^TYX = 30-year yield   → proxy for 30y
    Note: ^IRX is annualised discount rate, divide by 100 to get decimal.
    """
    print("  [yfinance] ^IRX ^FVX ^TYX ...", end=" ", flush=True)
    try:
        import yfinance as yf
        raw = yf.download(["^IRX", "^FVX", "^TYX"], start=START_DATE, end=END_DATE,
                          auto_adjust=True, progress=False)["Close"]
        # squeeze in case of MultiIndex
        if hasattr(raw.columns, 'levels'):
            raw = raw.droplevel(0, axis=1) if raw.columns.nlevels > 1 else raw
        raw = raw.dropna(how="all")
        if raw.empty or len(raw) < 10:
            raise ValueError("No data")
        print(f"OK ({len(raw)} rows)")
        result = {}
        if "^IRX" in raw.columns: result["r2"]  = raw["^IRX"].dropna()
        if "^FVX" in raw.columns: result["r5"]  = raw["^FVX"].dropna()
        if "^TYX" in raw.columns: result["r30"] = raw["^TYX"].dropna()
        return result if result else None
    except Exception as e:
        print(f"FAILED: {e}")
        return None

def fetch_treasury_yields():
    """
    US Treasury daily PAR yield curve (no API key required).
    Returns real 2y / 5y / 10y / 30y — a reliable substitute for FRED's
    DGS constant-maturity series, which time out on some networks.
    """
    import io, requests
    headers = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/124.0 Safari/537.36")}
    years  = sorted({int(START_DATE[:4]), int(END_DATE[:4])})
    frames = []
    for yr in years:
        url = ("https://home.treasury.gov/resource-center/data-chart-center/"
               "interest-rates/daily-treasury-rates.csv/{y}/all"
               "?type=daily_treasury_yield_curve&field_tdr_date_value={y}"
               "&page&_format=csv").format(y=yr)
        print(f"  [Treasury] par yields {yr} ...", end=" ", flush=True)
        try:
            r = requests.get(url, headers=headers, timeout=25)
            r.raise_for_status()
            d = pd.read_csv(io.StringIO(r.text))
            d.columns = d.columns.str.strip()
            frames.append(d)
            print(f"OK ({len(d)} rows)")
        except Exception as e:
            print(f"FAILED: {e}")
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    need = {"r2": "2 Yr", "r5": "5 Yr", "r10": "10 Yr", "r30": "30 Yr"}
    missing = [v for v in need.values() if v not in df.columns]
    if missing:
        print(f"  [Treasury] missing columns {missing}; got {list(df.columns)}")
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index().loc[START_DATE:END_DATE]
    out = {k: pd.to_numeric(df[v], errors="coerce").dropna() for k, v in need.items()}
    if any(len(s) < 10 for s in out.values()):
        print("  [Treasury] too few observations after date filtering")
        return None
    return out

def simulate_calibrated():
    print("  [simulation] Calibrated simulation.")
    dates = pd.date_range(START_DATE, END_DATE, freq="B")
    n = len(dates)
    rng = np.random.default_rng(42)

    def sim_oas(base, vol, drift):
        x = [base]
        for _ in range(n-1): x.append(max(20, x[-1] + rng.normal(drift, vol)))
        return pd.Series(x, index=dates)

    # Correlated curve: a shared Level factor + a Slope factor + small idiosyncratic
    # noise per tenor, so the PCA module recovers sensible Level/Slope/Curvature
    # shapes even when running offline on simulated data.
    lvl = np.cumsum(rng.normal(-0.004, 0.040, n))   # common level shock
    slp = np.cumsum(rng.normal( 0.000, 0.020, n))   # common slope shock
    def curve(base, b_slope, idio):
        noise = np.cumsum(rng.normal(0.0, idio, n))
        return pd.Series(np.maximum(0.5, base + lvl + b_slope*slp + noise), index=dates)

    return {
        "lqd_oas": sim_oas(108, 3.5, -0.06),
        "hyg_oas": sim_oas(345, 9.0, -0.10),
        "r2":  curve(4.80, -1.0, 0.012),
        "r5":  curve(4.42, -0.3, 0.010),
        "r10": curve(4.50,  0.3, 0.010),
        "r30": curve(4.65,  1.0, 0.012),
        "source": "calibrated simulation",
    }

# ── Attribution engine ─────────────────────────────────────────────────────────
def build_weekly(series):
    return series.resample("W-FRI").last().dropna()

def compute_attribution(oas_w, rates_w, duration, coupon, convexity,
                         krd_near, krd_far, near_key, far_key, **_):
    """
    KRD-based attribution:
      rate_near = −krd_near × Δr_near
      rate_far  = −krd_far  × Δr_far
      rate      = rate_near + rate_far
      spread    = −duration × ΔOAS
      carry     = coupon / 52
      convexity = 0.5 × C × (mean_dr + ΔOAS)²
    """
    dOAS     = oas_w.diff().dropna() / 10000
    dr_near  = rates_w[near_key].diff().dropna() / 100
    dr_far   = rates_w[far_key].diff().dropna() / 100

    common = dOAS.index.intersection(dr_near.index).intersection(dr_far.index)
    dOAS    = dOAS.loc[common]
    dr_near = dr_near.loc[common]
    dr_far  = dr_far.loc[common]

    carry      = coupon / 52
    spread     = -duration  * dOAS    * 100
    rate_near  = -krd_near  * dr_near * 100
    rate_far   = -krd_far   * dr_far  * 100
    rate_total = rate_near + rate_far

    # Convexity: use weighted average rate change
    dr_avg = (krd_near * dr_near + krd_far * dr_far) / (krd_near + krd_far + 1e-10)
    convex = 0.5 * convexity * (dr_avg + dOAS) ** 2 * 100

    return pd.DataFrame({
        "carry":      carry,
        "spread":     spread,
        "rate":       rate_total,
        "rate_near":  rate_near,
        "rate_far":   rate_far,
        "convexity":  convex,
        "total":      carry + spread + rate_total + convex,
    })

# ── Yield-curve PCA (Level / Slope / Curvature) ─────────────────────────────────
PCA_TENORS = [2, 5, 10, 30]
PCA_KEYS   = ["r2", "r5", "r10", "r30"]
_KEY_TENOR = {"r2": 2, "r5": 5, "r10": 10, "r30": 30}

def _sign_changes(v):
    s = np.sign(v); s = s[s != 0]
    return int(np.sum(s[:-1] * s[1:] < 0)) if len(s) > 1 else 0

def _orient(v, kind):
    """Eigenvector signs are arbitrary — fix them to a readable convention."""
    if kind == "level":
        if v.mean() < 0: v = -v
    elif kind == "slope":                       # long end positive → steepening = +
        if v[-1] < v[0]: v = -v
    elif kind == "curv":                         # wings positive, belly negative
        if (v[0] + v[-1]) - (v[1] + v[2]) < 0: v = -v
    return v

def compute_pca(rates_w, index):
    """
    PCA on weekly yield CHANGES at 2/5/10/30.
        covariance matrix  →  eigendecomposition  →  Level / Slope / Curvature.

    Scores are taken on the RAW (uncentred) weekly change, because the
    eigenvectors form a complete orthonormal basis of R^4 — so any move
    projects exactly, and  Σ_factor (−β × score)  reconciles with the KRD
    rate total to the penny (the residual is just PC4).
    """
    dY  = pd.DataFrame({k: rates_w[k].diff() for k in PCA_KEYS}).reindex(index).dropna()
    X   = dY.values                                      # N×4, percentage points
    cov = np.cov(X - X.mean(axis=0), rowvar=False)       # centred for estimation
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]                        # descending
    vals, vecs = vals[order], vecs[:, order]

    # Guard the slope/curv ordering by sign-change count (slope=1, curv=2)
    if _sign_changes(vecs[:, 1]) > _sign_changes(vecs[:, 2]):
        vecs[:, [1, 2]] = vecs[:, [2, 1]]
        vals[[1, 2]]    = vals[[2, 1]]

    var_ratio = (vals / vals.sum()).tolist()
    names = ["level", "slope", "curv", "resid"]
    vec = {nm: _orient(vecs[:, i].copy(), nm) for i, nm in enumerate(names)}

    scores = pd.DataFrame({nm: dY.values @ vec[nm] for nm in names}, index=dY.index)
    return {"vec": vec, "var": var_ratio, "scores": scores, "tenors": PCA_TENORS}

def krd_grid(params):
    """Map an ETF's two-node KRD onto the [2,5,10,30] grid (0 where it has no node)."""
    g = {2: 0.0, 5: 0.0, 10: 0.0, 30: 0.0}
    g[_KEY_TENOR[params["near_key"]]] += params["krd_near"]
    g[_KEY_TENOR[params["far_key"]]]  += params["krd_far"]
    return np.array([g[t] for t in PCA_TENORS])

# ── HTML ───────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Fixed Income Portfolio Attribution — Campisi Framework</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
:root{
  --bg:#0f1117;--bg2:#181c25;--bg3:#1e2330;--bdr:rgba(255,255,255,.08);
  --t:#e8eaf0;--t2:#8b90a0;--t3:#5a5f70;
  --ig:#185FA5;--hy:#993C1D;
  --rate:#534AB7;--rate2:#8B80E0;--spread:#1D9E75;--carry:#BA7517;--convex:#9E3D9E;
  --pos:#1D9E75;--neg:#C94040;--r:10px;
  --plot-bg:transparent;--plot-paper:transparent;--plot-grid:rgba(255,255,255,.05);
  --plot-tick:#8b90a0;--plot-line:rgba(255,255,255,.08);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
[data-theme="light"]{
  --bg:#f5f6fa;--bg2:#ffffff;--bg3:#eef0f5;--bdr:rgba(0,0,0,.08);
  --t:#1a1d24;--t2:#5a6070;--t3:#9aa0b0;
  --plot-grid:rgba(0,0,0,.06);--plot-tick:#5a6070;--plot-line:rgba(0,0,0,.1)}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--t);padding:2rem;min-height:100vh;transition:background .25s,color .25s}
.topbar{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1.5rem}
.topbar-left h1{font-size:18px;font-weight:500;margin-bottom:4px}
.sub{font-size:13px;color:var(--t2)}
.theme-btn{font-size:12px;padding:5px 12px;border-radius:var(--r);cursor:pointer;
  border:.5px solid var(--bdr);background:var(--bg2);color:var(--t);white-space:nowrap;
  transition:background .25s,color .25s}
.badge{font-size:11px;padding:2px 8px;border-radius:20px;margin-left:8px;vertical-align:middle}
.badge-live{background:rgba(29,158,117,.15);color:var(--pos)}
.badge-yf{background:rgba(83,74,183,.15);color:#534AB7}
.badge-sim{background:rgba(186,117,23,.15);color:#BA7517}
.formula{background:var(--bg2);border:.5px solid var(--bdr);border-radius:var(--r);
  padding:10px 16px;font-size:12px;color:var(--t2);margin-bottom:1.25rem;line-height:2.2}
.formula code{font-family:"SF Mono","Fira Code",monospace;color:var(--t);font-size:12px}
.krd-note{font-size:11px;color:var(--t3);margin-top:6px}
.weight-row{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.25rem}
.wc{background:var(--bg2);border:.5px solid var(--bdr);border-radius:var(--r);padding:1rem 1.25rem}
.wc-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.wc-name{font-size:14px;font-weight:500}
.wc-tag{font-size:11px;padding:2px 8px;border-radius:20px}
.wc-pct{font-size:22px;font-weight:500;margin:6px 0 8px}
.slider{width:100%;margin-bottom:10px}
.wc-meta{font-size:12px;color:var(--t2);margin-bottom:10px}
.ab{margin:5px 0}
.ab-row{display:flex;justify-content:space-between;font-size:12px;color:var(--t2);margin-bottom:3px}
.ab-track{height:4px;background:var(--bg3);border-radius:3px;overflow:hidden}
.ab-fill{height:100%;border-radius:3px;transition:width .35s}
.ab-sub{margin-left:10px}
.ab-sub .ab-track{height:3px;opacity:.6}
.metric-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:1.25rem}
.mc{background:var(--bg2);border:.5px solid var(--bdr);border-radius:var(--r);padding:12px 14px}
.mc-lbl{font-size:11px;color:var(--t2);margin-bottom:5px}
.mc-val{font-size:18px;font-weight:500}
.mc-sub{font-size:11px;margin-top:3px}
.pos{color:var(--pos)}.neg{color:var(--neg)}
.panel{background:var(--bg2);border:.5px solid var(--bdr);border-radius:var(--r);padding:1rem 1.25rem;margin-bottom:1rem}
.plbl{font-size:11px;font-weight:500;color:var(--t2);letter-spacing:.06em;text-transform:uppercase;margin-bottom:10px}
.tabs{display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap}
.tab{font-size:12px;padding:4px 12px;border-radius:var(--r);cursor:pointer;
  border:.5px solid var(--bdr);background:transparent;color:var(--t2)}
.tab.active{background:var(--bg3);color:var(--t);border-color:var(--t3)}
.legend{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:10px;font-size:12px;color:var(--t2)}
.ld{width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:4px;vertical-align:middle}
#chartDiv{width:100%;height:280px}
table{width:100%;border-collapse:collapse;font-size:12px}
th{font-size:11px;font-weight:500;color:var(--t2);text-align:left;padding:5px 8px;border-bottom:.5px solid var(--bdr)}
td{padding:6px 8px;border-bottom:.5px solid var(--bdr)}
tr:last-child td{border-bottom:none;font-weight:500}
.sub-row td{color:var(--t2);font-size:11px;padding:3px 8px 5px}
.sub-row td:first-child{padding-left:20px}
.mono{font-family:"SF Mono","Fira Code",monospace;font-size:11px;color:var(--t2)}
.note{font-size:11px;color:var(--t3);margin-top:.75rem}
.risk-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:1rem}
.rc{background:var(--bg3);border:.5px solid var(--bdr);border-radius:var(--r);padding:11px 13px}
.rc-lbl{font-size:11px;color:var(--t2);margin-bottom:5px}
.rc-val{font-size:17px;font-weight:500;color:var(--t)}
.rc-sub{font-size:11px;color:var(--t2);margin-top:2px}
.risk-bar{height:8px;background:var(--bg2);border-radius:4px;margin-top:8px;overflow:hidden}
.risk-bar-fill{height:100%;border-radius:4px}
</style>
</head>
<body>
<div class="topbar">
  <div class="topbar-left">
    <h1>Fixed income portfolio attribution <span class="badge __BADGE_CLASS__">__DATA_LABEL__</span></h1>
    <p class="sub">LQD (IG) &nbsp;·&nbsp; HYG (HY) &nbsp;·&nbsp; __START_DATE__ – __END_DATE__ &nbsp;·&nbsp; Campisi + KRD</p>
  </div>
  <button class="theme-btn" onclick="toggleTheme()" id="themeBtn">☀ Light</button>
</div>

<div class="formula">
  R = <code>Rate</code> + <code>Spread</code> + <code>Carry</code> + <code>Convexity</code><br>
  <code>Rate = −KRD<sub>near</sub> × Δr<sub>near</sub> − KRD<sub>far</sub> × Δr<sub>far</sub></code>
  &nbsp;·&nbsp; <code>Spread = −Duration × ΔOAS</code>
  &nbsp;·&nbsp; <code>Carry = coupon ÷ 52</code>
  &nbsp;·&nbsp; <code>Convexity = ½ · C · (Δr̄ + ΔOAS)²</code><br>
  <span class="krd-note">
    LQD nodes: KRD<sub>5y</sub> = __LQD_KRD_5__ &nbsp;·&nbsp; KRD<sub>30y</sub> = __LQD_KRD_30__
    &nbsp;&nbsp;|&nbsp;&nbsp;
    HYG nodes: KRD<sub>2y</sub> = __HYG_KRD_2__ &nbsp;·&nbsp; KRD<sub>5y</sub> = __HYG_KRD_5__
  </span>
</div>

<div class="weight-row">
  <div class="wc">
    <div class="wc-header">
      <span class="wc-name" style="color:var(--ig)">LQD</span>
      <span class="wc-tag" style="background:rgba(24,95,165,.12);color:var(--ig)">Investment grade · DGS5 + DGS30</span>
    </div>
    <div class="wc-pct" style="color:var(--ig)" id="lqdPct">60%</div>
    <input class="slider" type="range" min="0" max="100" value="60" id="lqdSlider" oninput="onSlider('lqd',this.value)">
    <div class="wc-meta">Dur: 8.4y · Coupon: ~4.8% · Convexity: 82</div>
    <div id="lqd-a-spread"    class="ab"></div>
    <div id="lqd-a-carry"     class="ab"></div>
    <div id="lqd-a-rate"      class="ab"></div>
    <div id="lqd-a-rate-near" class="ab ab-sub"></div>
    <div id="lqd-a-rate-far"  class="ab ab-sub"></div>
    <div id="lqd-a-convex"    class="ab"></div>
  </div>
  <div class="wc">
    <div class="wc-header">
      <span class="wc-name" style="color:var(--hy)">HYG</span>
      <span class="wc-tag" style="background:rgba(153,60,29,.12);color:var(--hy)">High yield · DGS2 + DGS5</span>
    </div>
    <div class="wc-pct" style="color:var(--hy)" id="hygPct">40%</div>
    <input class="slider" type="range" min="0" max="100" value="40" id="hygSlider" oninput="onSlider('hyg',this.value)">
    <div class="wc-meta">Dur: 4.1y · Coupon: ~7.2% · Convexity: 22</div>
    <div id="hyg-a-spread"    class="ab"></div>
    <div id="hyg-a-carry"     class="ab"></div>
    <div id="hyg-a-rate"      class="ab"></div>
    <div id="hyg-a-rate-near" class="ab ab-sub"></div>
    <div id="hyg-a-rate-far"  class="ab ab-sub"></div>
    <div id="hyg-a-convex"    class="ab"></div>
  </div>
</div>

<div class="metric-row" id="metrics"></div>

<div class="panel" id="attrPanel">
  <p class="plbl">Portfolio return decomposition</p>
  <div class="tabs">
    <button class="tab active" onclick="switchTab('cum',this)">Cumulative attribution</button>
    <button class="tab" onclick="switchTab('weekly',this)">Weekly stacked</button>
    <button class="tab" onclick="switchTab('oas',this)">OAS levels</button>
    <button class="tab" onclick="switchTab('curve',this)">Yield curve</button>
  </div>
  <div class="legend" id="legCum">
    <span><span class="ld" style="background:#d0d4e0"></span>Total (bold)</span>
    <span><span class="ld" style="background:var(--rate)"></span>Rate total</span>
    <span><span class="ld" style="background:var(--spread)"></span>Spread</span>
    <span><span class="ld" style="background:var(--carry)"></span>Carry</span>
    <span><span class="ld" style="background:var(--convex)"></span>Convexity</span>
    <span><span class="ld" style="background:var(--rate2);opacity:.8"></span>Rate near ···</span>
    <span><span class="ld" style="background:#9080FF;opacity:.8"></span>Rate far -·-</span>
  </div>
  <div class="legend" id="legOas" style="display:none">
    <span><span class="ld" style="background:var(--ig)"></span>LQD OAS (bp)</span>
    <span><span class="ld" style="background:var(--hy)"></span>HYG OAS (bp)</span>
  </div>
  <div class="legend" id="legCurve" style="display:none">
    <span><span class="ld" style="background:#4A9EFF"></span>2y yield</span>
    <span><span class="ld" style="background:var(--rate)"></span>5y yield</span>
    <span><span class="ld" style="background:#46b8a4"></span>10y yield</span>
    <span><span class="ld" style="background:#E05050"></span>30y yield</span>
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

<div class="panel" id="pcaPanel">
  <p class="plbl">Rate decomposition via yield-curve PCA — Level / Slope / Curvature</p>
  <p style="font-size:11px;color:var(--t3);margin:-4px 0 12px;line-height:1.6">
    Splits the <b>rate</b> component into three orthogonal curve factors estimated from DGS2/5/10/30 weekly changes.
    This measures the <i>shape</i> of the curve move — distinct from the bond-level
    <span style="color:var(--convex)">Convexity</span> term above. HYG's short duration means its curvature exposure is near zero.
  </p>
  <div class="risk-grid" id="pcaCards"></div>
  <div class="tabs" style="margin-top:1rem">
    <button class="tab active" onclick="switchPca('rate',this)">Rate split (L/S/C)</button>
    <button class="tab" onclick="switchPca('shapes',this)">Factor shapes</button>
  </div>
  <div class="legend" id="legPcaRate">
    <span><span class="ld" style="background:#e3a857"></span>Level</span>
    <span><span class="ld" style="background:#46b8a4"></span>Slope</span>
    <span><span class="ld" style="background:#a87fd6"></span>Curvature</span>
    <span><span class="ld" style="background:#d0d4e0"></span>Rate total (dashed)</span>
  </div>
  <div class="legend" id="legPcaShapes" style="display:none">
    <span><span class="ld" style="background:#e3a857"></span>Level (PC1)</span>
    <span><span class="ld" style="background:#46b8a4"></span>Slope (PC2)</span>
    <span><span class="ld" style="background:#a87fd6"></span>Curvature (PC3)</span>
  </div>
  <div id="pcaChart" style="width:100%;height:260px"></div>
</div>

<div class="panel" id="riskPanel">
  <p class="plbl">Risk metrics</p>
  <div class="tabs">
    <button class="tab active" onclick="switchRisk('snapshot',this)">Point-in-time</button>
    <button class="tab" onclick="switchRisk('portvar',this)">Portfolio VaR</button>
    <button class="tab" onclick="switchRisk('dts',this)">Rolling DTS</button>
    <button class="tab" onclick="switchRisk('dd',this)">Drawdown</button>
  </div>
  <div id="riskSnapshot">
    <div class="risk-grid" id="riskCards"></div>
    <table style="margin-top:1rem" id="riskTable">
      <thead><tr>
        <th>Metric</th><th>Formula</th><th>LQD</th><th>HYG</th><th>Portfolio (60/40)</th>
      </tr></thead>
      <tbody id="riskTbody"></tbody>
    </table>
  </div>
  <div id="riskChart" style="display:none;width:100%;height:220px"></div>
</div>

<div class="panel" style="margin-top:.5rem">
  <p class="plbl">Data provenance</p>
  <div id="infoPanel"></div>
</div>

<script>
const DATA = __DATA_JSON__;
let wLQD=0.60, wHYG=0.40, currentTab='cum';

function sum(arr,k){return arr.reduce((a,b)=>a+b[k],0);}

function getTotals(){
  const KEYS=['spread','carry','rate','rate_near','rate_far',
              'rate_level','rate_slope','rate_curv','rate_resid','convexity'];
  const L={},H={};
  KEYS.forEach(k=>{L[k]=sum(DATA.lqd,k);H[k]=sum(DATA.hyg,k);});
  const mk=(o,w)=>{const r={};KEYS.forEach(k=>r[k]=w*o[k]);
    r.total=w*(o.spread+o.carry+o.rate+o.convexity);return r;};
  const wl=mk(L,wLQD), wh=mk(H,wHYG), port={};
  Object.keys(wl).forEach(k=>port[k]=wl[k]+wh[k]);
  return {wlqd:wl, whyg:wh, port:port};
}

function makeBar(id,label,val,color,small){
  const h=small?3:4;
  const pct=Math.min(100,Math.abs(val)/3*100);
  const vc=val>=0?'var(--pos)':'var(--neg)';
  const sign=val>=0?'+':'';
  document.getElementById(id).innerHTML=
    '<div class="ab-row"><span style="font-size:'+(small?'11':'12')+'px;color:var(--t2)">'+label+'</span>'+
    '<span style="color:'+vc+';font-size:'+(small?'11':'12')+'px">'+sign+val.toFixed(2)+'%</span></div>'+
    '<div class="ab-track" style="height:'+h+'px">'+
    '<div class="ab-fill" style="width:'+pct+'%;background:'+color+'"></div></div>';
}

function updateCards(t){
  document.getElementById('lqdPct').textContent=Math.round(wLQD*100)+'%';
  document.getElementById('hygPct').textContent=Math.round(wHYG*100)+'%';
  makeBar('lqd-a-spread',   'Spread',              t.wlqd.spread,    'var(--spread)');
  makeBar('lqd-a-carry',    'Carry',               t.wlqd.carry,     'var(--carry)');
  makeBar('lqd-a-rate',     'Rate (total)',         t.wlqd.rate,      'var(--rate)');
  makeBar('lqd-a-rate-near','  ↳ 5y node',          t.wlqd.rate_near, 'var(--rate2)',true);
  makeBar('lqd-a-rate-far', '  ↳ 30y node',         t.wlqd.rate_far,  '#9080FF',    true);
  makeBar('lqd-a-convex',   'Convexity',            t.wlqd.convexity, 'var(--convex)');
  makeBar('hyg-a-spread',   'Spread',              t.whyg.spread,    'var(--spread)');
  makeBar('hyg-a-carry',    'Carry',               t.whyg.carry,     'var(--carry)');
  makeBar('hyg-a-rate',     'Rate (total)',         t.whyg.rate,      'var(--rate)');
  makeBar('hyg-a-rate-near','  ↳ 2y node',          t.whyg.rate_near, 'var(--rate2)',true);
  makeBar('hyg-a-rate-far', '  ↳ 5y node',          t.whyg.rate_far,  '#9080FF',    true);
  makeBar('hyg-a-convex',   'Convexity',            t.whyg.convexity, 'var(--convex)');
}

function updateMetrics(t){
  const N=DATA.lqd_oas.length;
  const oasNow=wLQD*DATA.lqd_oas[N-1]+wHYG*DATA.hyg_oas[N-1];
  const oasChg=oasNow-(wLQD*DATA.lqd_oas[0]+wHYG*DATA.hyg_oas[0]);
  const items=[
    {l:'Portfolio total',   v:(t.port.total>=0?'+':'')+t.port.total.toFixed(2)+'%',      s:'All components',   p:t.port.total>=0},
    {l:'Wtd avg OAS',       v:oasNow.toFixed(0)+' bp',
     s:(oasChg>=0?'+':'')+oasChg.toFixed(0)+' bp vs start',                              p:oasChg<0},
    {l:'Spread contrib',    v:(t.port.spread>=0?'+':'')+t.port.spread.toFixed(2)+'%',    s:'−Dur × ΔOAS',      p:t.port.spread>=0},
    {l:'Rate contrib',      v:(t.port.rate>=0?'+':'')+t.port.rate.toFixed(2)+'%',        s:'KRD near+far',     p:t.port.rate>=0},
    {l:'  Near node',       v:(t.port.rate_near>=0?'+':'')+t.port.rate_near.toFixed(2)+'%', s:'LQD:5y  HYG:2y', p:t.port.rate_near>=0},
    {l:'  Far node',        v:(t.port.rate_far>=0?'+':'')+t.port.rate_far.toFixed(2)+'%',   s:'LQD:30y HYG:5y', p:t.port.rate_far>=0},
    {l:'Carry',             v:'+'+t.port.carry.toFixed(2)+'%',                           s:'Coupon income',    p:true},
    {l:'Convexity',         v:(t.port.convexity>=0?'+':'')+t.port.convexity.toFixed(2)+'%', s:'½·C·(Δr̄+ΔOAS)²', p:t.port.convexity>=0},
  ];
  document.getElementById('metrics').innerHTML=items.map(function(m){
    return '<div class="mc">'+
      '<div class="mc-lbl">'+m.l+'</div>'+
      '<div class="mc-val '+(m.p?'pos':'neg')+'">'+m.v+'</div>'+
      '<div class="mc-sub '+(m.p?'pos':'neg')+'">'+m.s+'</div>'+
      '</div>';
  }).join('');
}

function updateTable(t){
  const rows=[
    {c:'Rate contribution', f:'KRD×Δr near+far', lqd:t.wlqd.rate,      hyg:t.whyg.rate,      p:t.port.rate,      sub:true},
    {c:'  Near node',       f:'LQD:−KRD₅×Δr₅ / HYG:−KRD₂×Δr₂',
                                                  lqd:t.wlqd.rate_near, hyg:t.whyg.rate_near, p:t.port.rate_near, sub:false, indent:true},
    {c:'  Far node',        f:'LQD:−KRD₃₀×Δr₃₀ / HYG:−KRD₅×Δr₅',
                                                  lqd:t.wlqd.rate_far,  hyg:t.whyg.rate_far,  p:t.port.rate_far,  sub:false, indent:true},
    {c:'Spread contribution',f:'−Dur × ΔOAS',    lqd:t.wlqd.spread,    hyg:t.whyg.spread,    p:t.port.spread},
    {c:'Carry',              f:'coupon ÷ 52',     lqd:t.wlqd.carry,     hyg:t.whyg.carry,     p:t.port.carry},
    {c:'Convexity',          f:'½·C·(Δr̄+ΔOAS)²', lqd:t.wlqd.convexity, hyg:t.whyg.convexity, p:t.port.convexity},
    {c:'Total return',       f:'Sum',             lqd:t.wlqd.total,     hyg:t.whyg.total,     p:t.port.total,     total:true},
  ];
  const topSep='border-top:.5px solid rgba(255,255,255,.08);';
  document.getElementById('summaryTbody').innerHTML=rows.map((r,i)=>{
    const s=r.total?topSep:'';
    const dim=r.indent?'color:var(--t2);font-size:11px;':'';
    const c=v=>v>=0?'var(--pos)':'var(--neg)';
    const f=v=>(v>=0?'+':'')+v.toFixed(2)+'%';
    const pl=r.indent?'padding-left:20px;':'';
    return '<tr>'+
      '<td style="'+s+dim+pl+'">'+r.c+'</td>'+
      '<td class="mono" style="'+s+dim+'">'+r.f+'</td>'+
      '<td style="color:'+c(r.lqd)+';'+s+dim+'">'+f(r.lqd)+'</td>'+
      '<td style="color:'+c(r.hyg)+';'+s+dim+'">'+f(r.hyg)+'</td>'+
      '<td style="color:'+c(r.p)+';'+s+'">'+f(r.p)+'</td>'+
      '</tr>';
  }).join('');
}

const C={rate:'#534AB7',rate2:'#8B80E0',spread:'#1D9E75',carry:'#BA7517',convex:'#9E3D9E',total:'#888',ig:'#185FA5',hy:'#993C1D'};

function getLY() {
  const dark = document.documentElement.getAttribute('data-theme') !== 'light';
  return {
    paper_bgcolor:'transparent', plot_bgcolor:'transparent',
    margin:{l:44,r:20,t:10,b:40},
    font:{family:'-apple-system,sans-serif',size:11,color:dark?'#8b90a0':'#5a6070'},
    xaxis:{showgrid:false,tickcolor:'transparent',linecolor:dark?'rgba(255,255,255,.08)':'rgba(0,0,0,.1)'},
    yaxis:{gridcolor:dark?'rgba(255,255,255,.05)':'rgba(0,0,0,.06)',
           tickcolor:'transparent',linecolor:dark?'rgba(255,255,255,.08)':'rgba(0,0,0,.1)'},
    legend:{bgcolor:'transparent'},hovermode:'x unified',
  };
}

function toggleTheme() {
  const html = document.documentElement;
  const isDark = html.getAttribute('data-theme') === 'dark';
  html.setAttribute('data-theme', isDark ? 'light' : 'dark');
  document.getElementById('themeBtn').textContent = isDark ? '🌙 Dark' : '☀ Light';
  // Redraw all active charts with new theme
  redraw();
  if(currentRiskTab !== 'snapshot') {
    if(currentRiskTab==='portvar') buildPortVarChart();
    else if(currentRiskTab==='dts') buildDTSChart();
    else buildDrawdownChart();
  }
}

function buildPortVarChart() {
  const R = DATA.risk;
  // Current weight position
  const curX = Math.round(wLQD * 100);
  const curY = R.var_curve_y[Math.round(curX/5)] || R.port_var_60_40;

  Plotly.newPlot('riskChart',[
    {x:R.var_curve_x, y:R.var_naive_y, name:'Naive (no diversif.)', mode:'lines',
     line:{color:'#888',width:1.5,dash:'dot'},
     hovertemplate:'LQD %{x}%: %{y:.2f}%<extra>No diversification</extra>'},
    {x:R.var_curve_x, y:R.var_curve_y, name:'Portfolio VaR (with corr)', mode:'lines',
     line:{color:C.rate,width:2},
     fill:'tonexty', fillcolor:'rgba(83,74,183,.08)',
     hovertemplate:'LQD %{x}%: %{y:.2f}%<extra>With correlation</extra>'},
    {x:[curX], y:[curY], name:'Current weights', mode:'markers',
     marker:{color:C.ig,size:10,symbol:'circle'},
     hovertemplate:'LQD '+curX+'%: '+curY.toFixed(2)+'%<extra>Current</extra>'},
  ],{...getLY(),
    xaxis:{...getLY().xaxis,title:{text:'LQD weight (%)',font:{size:11}},ticksuffix:'%'},
    yaxis:{...getLY().yaxis,ticksuffix:'%',title:{text:'10-day Spread VaR (95%)',font:{size:11}}},
    annotations:[{
      x:curX, y:curY, xref:'x', yref:'y',
      text:'  '+curX+'/'+Math.round(wHYG*100)+' split<br>  VaR='+curY.toFixed(2)+'%<br>  Div benefit='+R.divers_benefit.toFixed(2)+'%',
      showarrow:true, arrowhead:2, arrowcolor:C.ig,
      font:{size:11,color:C.ig}, bgcolor:'transparent', bordercolor:'transparent', ax:40, ay:-40,
    }],
  },{responsive:true,displayModeBar:false});
}

function mkLine(name,x,y,color,dash,width){
  return{x,y,name,mode:'lines',line:{color,width:width||1.5,dash:dash||'solid'},
    hovertemplate:'%{y:.2f}%<extra>'+name+'</extra>'};
}
function mkBar(name,x,y,color){
  return{x,y,name,type:'bar',marker:{color},hovertemplate:'%{y:.3f}%<extra>'+name+'</extra>'};
}

function buildCumChart(){
  let cr=0,crn=0,crf=0,cs=0,cc=0,cx=0,ct=0;
  const R=[],RN=[],RF=[],S=[],Ca=[],X=[],T=[];
  DATA.lqd.forEach((_,i)=>{
    cr+=wLQD*DATA.lqd[i].rate+wHYG*DATA.hyg[i].rate;
    crn+=wLQD*DATA.lqd[i].rate_near+wHYG*DATA.hyg[i].rate_near;
    crf+=wLQD*DATA.lqd[i].rate_far+wHYG*DATA.hyg[i].rate_far;
    cs+=wLQD*DATA.lqd[i].spread+wHYG*DATA.hyg[i].spread;
    cc+=wLQD*DATA.lqd[i].carry+wHYG*DATA.hyg[i].carry;
    cx+=wLQD*DATA.lqd[i].convexity+wHYG*DATA.hyg[i].convexity;
    ct+=wLQD*DATA.lqd[i].total+wHYG*DATA.hyg[i].total;
    R.push(+cr.toFixed(3));RN.push(+crn.toFixed(3));RF.push(+crf.toFixed(3));
    S.push(+cs.toFixed(3));Ca.push(+cc.toFixed(3));X.push(+cx.toFixed(3));T.push(+ct.toFixed(3));
  });
  Plotly.newPlot('chartDiv',[
    // Main components — solid medium lines
    {x:DATA.labels,y:R, name:'Rate (total)', mode:'lines',
     line:{color:C.rate,  width:1.8,dash:'solid'}, hovertemplate:'%{y:.2f}%<extra>Rate total</extra>'},
    {x:DATA.labels,y:S, name:'Spread',       mode:'lines',
     line:{color:C.spread,width:1.8,dash:'solid'}, hovertemplate:'%{y:.2f}%<extra>Spread</extra>'},
    {x:DATA.labels,y:Ca,name:'Carry',        mode:'lines',
     line:{color:C.carry, width:1.8,dash:'solid'}, hovertemplate:'%{y:.2f}%<extra>Carry</extra>'},
    {x:DATA.labels,y:X, name:'Convexity',    mode:'lines',
     line:{color:C.convex,width:1.8,dash:'solid'}, hovertemplate:'%{y:.2f}%<extra>Convexity</extra>'},
    // KRD sub-components — thin dashed lines
    {x:DATA.labels,y:RN,name:'Rate near node',mode:'lines',
     line:{color:C.rate2, width:0.9,dash:'dot'},     opacity:0.8,
     hovertemplate:'%{y:.2f}%<extra>Rate near</extra>'},
    {x:DATA.labels,y:RF,name:'Rate far node', mode:'lines',
     line:{color:'#9080FF',width:0.9,dash:'dashdot'}, opacity:0.8,
     hovertemplate:'%{y:.2f}%<extra>Rate far</extra>'},
    // Total — bold solid white-ish
    {x:DATA.labels,y:T, name:'Total',         mode:'lines',
     line:{color:'#d0d4e0',width:2.5,dash:'solid'},
     hovertemplate:'%{y:.2f}%<extra>Total</extra>'},
  ],{...getLY(),yaxis:{...getLY().yaxis,ticksuffix:'%'}},{responsive:true,displayModeBar:false});
}

function buildWeeklyChart(){
  const pR=[],pS=[],pC=[],pX=[];
  DATA.lqd.forEach((_,i)=>{
    pR.push(+(wLQD*DATA.lqd[i].rate+wHYG*DATA.hyg[i].rate).toFixed(4));
    pS.push(+(wLQD*DATA.lqd[i].spread+wHYG*DATA.hyg[i].spread).toFixed(4));
    pC.push(+(wLQD*DATA.lqd[i].carry+wHYG*DATA.hyg[i].carry).toFixed(4));
    pX.push(+(wLQD*DATA.lqd[i].convexity+wHYG*DATA.hyg[i].convexity).toFixed(4));
  });
  Plotly.newPlot('chartDiv',[
    mkBar('Rate',      DATA.labels,pR,'rgba(83,74,183,.75)'),
    mkBar('Spread',    DATA.labels,pS,'rgba(29,158,117,.75)'),
    mkBar('Carry',     DATA.labels,pC,'rgba(186,117,23,.75)'),
    mkBar('Convexity', DATA.labels,pX,'rgba(158,61,158,.75)'),
  ],{...getLY(),barmode:'stack',yaxis:{...getLY().yaxis,ticksuffix:'%'}},{responsive:true,displayModeBar:false});
}

function buildOASChart(){
  Plotly.newPlot('chartDiv',[
    {x:DATA.oas_dates,y:DATA.lqd_oas,name:'LQD OAS',mode:'lines',
     line:{color:C.ig,width:1.5},hovertemplate:'%{y:.0f} bp<extra>LQD</extra>'},
    {x:DATA.oas_dates,y:DATA.hyg_oas,name:'HYG OAS',mode:'lines',
     line:{color:C.hy,width:1.5},yaxis:'y2',hovertemplate:'%{y:.0f} bp<extra>HYG</extra>'},
  ],{...getLY(),
    yaxis:{...getLY().yaxis,title:{text:'LQD OAS (bp)',font:{color:C.ig,size:11}}},
    yaxis2:{overlaying:'y',side:'right',gridcolor:'transparent',tickcolor:'transparent',
            linecolor:'rgba(255,255,255,.08)',title:{text:'HYG OAS (bp)',font:{color:C.hy,size:11}}},
  },{responsive:true,displayModeBar:false});
}

function buildCurveChart(){
  const traces=[];
  if(DATA.r2) traces.push({x:DATA.oas_dates,y:DATA.r2,name:'2y yield',mode:'lines',
    line:{color:'#4A9EFF',width:1.5},hovertemplate:'%{y:.2f}%<extra>2y</extra>'});
  if(DATA.r5) traces.push({x:DATA.oas_dates,y:DATA.r5,name:'5y yield',mode:'lines',
    line:{color:C.rate,width:1.5},hovertemplate:'%{y:.2f}%<extra>5y</extra>'});
  if(DATA.r10) traces.push({x:DATA.oas_dates,y:DATA.r10,name:'10y yield',mode:'lines',
    line:{color:'#46b8a4',width:1.5},hovertemplate:'%{y:.2f}%<extra>10y</extra>'});
  if(DATA.r30) traces.push({x:DATA.oas_dates,y:DATA.r30,name:'30y yield',mode:'lines',
    line:{color:'#E05050',width:1.5},hovertemplate:'%{y:.2f}%<extra>30y</extra>'});
  Plotly.newPlot('chartDiv',traces,
    {...getLY(),yaxis:{...getLY().yaxis,ticksuffix:'%'}},{responsive:true,displayModeBar:false});
}

// ── Risk panel ────────────────────────────────────────────────────────────────
let currentRiskTab = 'snapshot';

function portRisk(w_lqd, w_hyg) {
  const R = DATA.risk;
  const dts   = w_lqd*R.lqd.dts   + w_hyg*R.hyg.dts;
  const dv01  = w_lqd*R.lqd.dv01  + w_hyg*R.hyg.dv01;
  const svar  = w_lqd*R.lqd.svar_10d  + w_hyg*R.hyg.svar_10d;
  const scvar = w_lqd*R.lqd.scvar_10d + w_hyg*R.hyg.scvar_10d;
  const rc_lqd = w_lqd*R.lqd.dts / (w_lqd*R.lqd.dts + w_hyg*R.hyg.dts + 1e-10);
  return { dts, dv01, svar, scvar, rc_lqd: rc_lqd*100, rc_hyg: (1-rc_lqd)*100 };
}

function updateRiskPanel() {
  const R = DATA.risk;
  const p = portRisk(wLQD, wHYG);

  // Metric cards
  const cards = [
    { l:'Portfolio DTS',       v:p.dts.toFixed(0)+' bp·yr', s:'Duration × OAS' },
    { l:'Portfolio DV01',      v:'$'+p.dv01.toFixed(0),     s:'Per $1M notional' },
    { l:'Spread VaR (10d 95%)',v:p.svar.toFixed(2)+'%',     s:'Historical simulation' },
    { l:'Spread CVaR (10d)',   v:p.scvar.toFixed(2)+'%',    s:'Expected shortfall' },
    { l:'IG risk contrib',     v:p.rc_lqd.toFixed(1)+'%',   s:'DTS-weighted' },
    { l:'HY risk contrib',     v:p.rc_hyg.toFixed(1)+'%',   s:'DTS-weighted' },
  ];
  document.getElementById('riskCards').innerHTML = cards.map(function(c){
    return '<div class="rc">'+
      '<div class="rc-lbl">'+c.l+'</div>'+
      '<div class="rc-val">'+c.v+'</div>'+
      '<div class="rc-sub">'+c.s+'</div>'+
      '</div>';
  }).join('');

  // Risk contribution bar
  document.getElementById('riskCards').innerHTML +=
    '<div class="rc" style="grid-column:1/-1">'+
      '<div class="rc-lbl">Risk contribution split (DTS-weighted) — IG vs HY</div>'+
      '<div style="display:flex;gap:8px;align-items:center;margin-top:6px">'+
        '<span style="font-size:12px;color:var(--ig);width:40px">'+p.rc_lqd.toFixed(1)+'%</span>'+
        '<div style="flex:1;height:10px;background:var(--bg3);border-radius:5px;overflow:hidden">'+
          '<div style="width:'+p.rc_lqd+'%;height:100%;background:linear-gradient(90deg,var(--ig),var(--hy));border-radius:5px"></div>'+
        '</div>'+
        '<span style="font-size:12px;color:var(--hy);width:40px;text-align:right">'+p.rc_hyg.toFixed(1)+'%</span>'+
      '</div>'+
      '<div style="display:flex;justify-content:space-between;font-size:10px;color:var(--t3);margin-top:2px">'+
        '<span>LQD (IG)</span><span>HYG (HY)</span>'+
      '</div>'+
    '</div>';

  // Risk table
  const rows = [
    { m:'DTS',               f:'Dur × OAS',           lqd:R.lqd.dts.toFixed(0)+' bp·yr',  hyg:R.hyg.dts.toFixed(0)+' bp·yr',  port:p.dts.toFixed(0)+' bp·yr' },
    { m:'DV01 (total)',      f:'Dur × 0.0001 × $10M', lqd:'$'+R.lqd.dv01.toLocaleString(), hyg:'$'+R.hyg.dv01.toLocaleString(), port:'$'+p.dv01.toFixed(0) },
    { m:'  DV01 near node',  f:'KRD_near × 0.0001',   lqd:'$'+R.lqd.dv01_5+' (5y)',       hyg:'$'+R.hyg.dv01_2+' (2y)',        port:'—', indent:true },
    { m:'  DV01 far node',   f:'KRD_far × 0.0001',    lqd:'$'+R.lqd.dv01_30+' (30y)',     hyg:'$'+R.hyg.dv01_5+' (5y)',        port:'—', indent:true },
    { m:'Spread VaR 10d 95%',f:'Hist sim √(10/5)',    lqd:R.lqd.svar_10d.toFixed(2)+'%',  hyg:R.hyg.svar_10d.toFixed(2)+'%',  port:R.port_var_60_40.toFixed(2)+'%' },
    { m:'Spread CVaR 10d',   f:'E[loss | > VaR]',     lqd:R.lqd.scvar_10d.toFixed(2)+'%', hyg:R.hyg.scvar_10d.toFixed(2)+'%', port:p.scvar.toFixed(2)+'%' },
    { m:'Correlation',       f:'LQD-HYG spread P&L',  lqd:'—',                             hyg:'—',                             port:R.corr.toFixed(3) },
    { m:'Diversif. benefit', f:'Naive − portfolio VaR',lqd:'—',                             hyg:'—',                             port:R.divers_benefit.toFixed(2)+'%' },
    { m:'Max drawdown',      f:'Peak-to-trough',       lqd:R.lqd.mdd.toFixed(2)+'%',       hyg:R.hyg.mdd.toFixed(2)+'%',       port:'—' },
  ];
  const sep='border-top:.5px solid rgba(255,255,255,.08);';
  document.getElementById('riskTbody').innerHTML = rows.map((r,i)=>{
    const dim = r.indent ? 'color:var(--t2);font-size:11px;' : '';
    const pl  = r.indent ? 'padding-left:18px;' : '';
    return '<tr>'+
      '<td style="'+dim+pl+'">'+r.m+'</td>'+
      '<td class="mono" style="'+dim+'">'+r.f+'</td>'+
      '<td style="'+dim+'">'+r.lqd+'</td>'+
      '<td style="'+dim+'">'+r.hyg+'</td>'+
      '<td>'+r.port+'</td>'+
      '</tr>';
  }).join('');
}

function buildDTSChart() {
  const R = DATA.risk;
  const portDTS = DATA.risk.rolling_lqd_dts.map((v,i)=>
    +(wLQD*v + wHYG*R.rolling_hyg_dts[i]).toFixed(1));
  Plotly.newPlot('riskChart',[
    {x:DATA.oas_dates, y:R.rolling_lqd_dts, name:'LQD DTS', mode:'lines',
     line:{color:C.ig,width:1.5}, hovertemplate:'%{y:.0f} bp·yr<extra>LQD</extra>'},
    {x:DATA.oas_dates, y:R.rolling_hyg_dts, name:'HYG DTS', mode:'lines',
     line:{color:C.hy,width:1.5}, yaxis:'y2', hovertemplate:'%{y:.0f} bp·yr<extra>HYG</extra>'},
    {x:DATA.oas_dates, y:portDTS, name:'Portfolio DTS', mode:'lines',
     line:{color:'#d0d4e0',width:2,dash:'dash'}, hovertemplate:'%{y:.0f} bp·yr<extra>Portfolio</extra>'},
  ],{...getLY(),
    yaxis: {...getLY().yaxis,title:{text:'LQD DTS (bp·yr)',font:{color:C.ig,size:11}}},
    yaxis2:{overlaying:'y',side:'right',gridcolor:'transparent',tickcolor:'transparent',
            linecolor:'rgba(255,255,255,.08)',title:{text:'HYG DTS (bp·yr)',font:{color:C.hy,size:11}}},
  },{responsive:true,displayModeBar:false});
}

function buildDrawdownChart() {
  const R = DATA.risk;
  const portDD = R.lqd_dd.map((v,i) => +(wLQD*v + wHYG*R.hyg_dd[i]).toFixed(3));
  Plotly.newPlot('riskChart',[
    {x:DATA.labels, y:R.lqd_dd, name:'LQD', mode:'lines', fill:'tozeroy',
     line:{color:C.ig,width:1}, fillcolor:'rgba(24,95,165,.15)',
     hovertemplate:'%{y:.2f}%<extra>LQD</extra>'},
    {x:DATA.labels, y:R.hyg_dd, name:'HYG', mode:'lines', fill:'tozeroy',
     line:{color:C.hy,width:1}, fillcolor:'rgba(153,60,29,.15)',
     hovertemplate:'%{y:.2f}%<extra>HYG</extra>'},
    {x:DATA.labels, y:portDD, name:'Portfolio', mode:'lines',
     line:{color:'#d0d4e0',width:2,dash:'dash'},
     hovertemplate:'%{y:.2f}%<extra>Portfolio</extra>'},
  ],{...getLY(),yaxis:{...getLY().yaxis,ticksuffix:'%'}},{responsive:true,displayModeBar:false});
}

function switchRisk(tab, btn) {
  currentRiskTab = tab;
  document.getElementById('riskSnapshot').style.display = 'none';
  document.getElementById('riskChart').style.display    = 'none';

  document.querySelectorAll('#riskPanel .tab').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');

  if (tab === 'snapshot') {
    document.getElementById('riskSnapshot').style.display = 'block';
    updateRiskPanel();
  } else {
    document.getElementById('riskChart').style.display = 'block';
    if      (tab === 'portvar') buildPortVarChart();
    else if (tab === 'dts')     buildDTSChart();
    else                        buildDrawdownChart();
  }
}

const PCA_COL={level:'#e3a857',slope:'#46b8a4',curv:'#a87fd6'};
let currentPcaTab='rate';

function updatePcaCards(t){
  const P=DATA.pca, bL=P.beta_lqd, bH=P.beta_hyg;
  const pct=x=>(x*100).toFixed(1)+'%';
  const sgn=x=>(x>=0?'+':'')+x.toFixed(2)+'%';
  const cards=[
    {l:'Variance explained',      v:pct(P.var[0])+' / '+pct(P.var[1])+' / '+pct(P.var[2]), s:'Level / Slope / Curv'},
    {l:'LQD factor duration',     v:bL.level.toFixed(2)+' / '+bL.slope.toFixed(2)+' / '+bL.curv.toFixed(2), s:'L / S / C  (KRD·v)'},
    {l:'HYG factor duration',     v:bH.level.toFixed(2)+' / '+bH.slope.toFixed(2)+' / '+bH.curv.toFixed(2), s:'short-dur → curv≈0'},
    {l:'Portfolio rate · Level',  v:sgn(t.port.rate_level), s:'−β_L × score'},
    {l:'Portfolio rate · Slope',  v:sgn(t.port.rate_slope), s:'−β_S × score'},
    {l:'Portfolio rate · Curv',   v:sgn(t.port.rate_curv),  s:'−β_C × score'},
  ];
  document.getElementById('pcaCards').innerHTML=cards.map(c=>
    '<div class="rc"><div class="rc-lbl">'+c.l+'</div>'+
    '<div class="rc-val">'+c.v+'</div><div class="rc-sub">'+c.s+'</div></div>').join('');
}

function buildPcaRateChart(){
  let cl=0,cs=0,cc=0,ct=0; const L=[],S=[],Cv=[],T=[];
  DATA.lqd.forEach((_,i)=>{
    cl+=wLQD*DATA.lqd[i].rate_level+wHYG*DATA.hyg[i].rate_level;
    cs+=wLQD*DATA.lqd[i].rate_slope+wHYG*DATA.hyg[i].rate_slope;
    cc+=wLQD*DATA.lqd[i].rate_curv +wHYG*DATA.hyg[i].rate_curv;
    ct+=wLQD*DATA.lqd[i].rate      +wHYG*DATA.hyg[i].rate;
    L.push(+cl.toFixed(3));S.push(+cs.toFixed(3));Cv.push(+cc.toFixed(3));T.push(+ct.toFixed(3));
  });
  Plotly.newPlot('pcaChart',[
    {x:DATA.labels,y:L, name:'Level',     mode:'lines',line:{color:PCA_COL.level,width:1.8},hovertemplate:'%{y:.2f}%<extra>Level</extra>'},
    {x:DATA.labels,y:S, name:'Slope',     mode:'lines',line:{color:PCA_COL.slope,width:1.8},hovertemplate:'%{y:.2f}%<extra>Slope</extra>'},
    {x:DATA.labels,y:Cv,name:'Curvature', mode:'lines',line:{color:PCA_COL.curv, width:1.8},hovertemplate:'%{y:.2f}%<extra>Curvature</extra>'},
    {x:DATA.labels,y:T, name:'Rate total',mode:'lines',line:{color:'#d0d4e0',width:2.4,dash:'dash'},hovertemplate:'%{y:.2f}%<extra>Rate total</extra>'},
  ],{...getLY(),yaxis:{...getLY().yaxis,ticksuffix:'%'}},{responsive:true,displayModeBar:false});
}

function buildPcaShapesChart(){
  const P=DATA.pca, x=P.tenors.map(t=>t+'y'), ly=getLY();
  Plotly.newPlot('pcaChart',[
    {x:x,y:P.vec_level,name:'Level',    mode:'lines+markers',line:{color:PCA_COL.level,width:2},marker:{size:7},hovertemplate:'%{y:.2f}<extra>Level</extra>'},
    {x:x,y:P.vec_slope,name:'Slope',    mode:'lines+markers',line:{color:PCA_COL.slope,width:2},marker:{size:7},hovertemplate:'%{y:.2f}<extra>Slope</extra>'},
    {x:x,y:P.vec_curv, name:'Curvature',mode:'lines+markers',line:{color:PCA_COL.curv, width:2},marker:{size:7},hovertemplate:'%{y:.2f}<extra>Curvature</extra>'},
  ],{...ly,margin:{l:44,r:20,t:10,b:36},
     xaxis:{...ly.xaxis,title:{text:'Maturity',font:{size:11}}},
     yaxis:{...ly.yaxis,title:{text:'Eigenvector weight',font:{size:11}},zeroline:true,zerolinecolor:'rgba(128,128,128,.35)'}},
   {responsive:true,displayModeBar:false});
}

function switchPca(tab,btn){
  currentPcaTab=tab;
  document.querySelectorAll('#pcaPanel .tab').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('legPcaRate').style.display   = tab==='rate'  ?'flex':'none';
  document.getElementById('legPcaShapes').style.display = tab==='shapes'?'flex':'none';
  if(tab==='rate') buildPcaRateChart(); else buildPcaShapesChart();
}

function redraw() {
  const t=getTotals();
  updateCards(t);updateMetrics(t);updateTable(t);
  updateRiskPanel();
  updatePcaCards(t);
  if(currentTab==='cum')buildCumChart();
  else if(currentTab==='weekly')buildWeeklyChart();
  else if(currentTab==='oas')buildOASChart();
  else buildCurveChart();
  if(currentPcaTab==='rate')buildPcaRateChart();else buildPcaShapesChart();
}

function onSlider(which,val){
  const v=parseInt(val)/100;
  if(which==='lqd'){wLQD=v;wHYG=Math.max(0,1-v);document.getElementById('hygSlider').value=Math.round(wHYG*100);}
  else{wHYG=v;wLQD=Math.max(0,1-v);document.getElementById('lqdSlider').value=Math.round(wLQD*100);}
  redraw();
  if(currentRiskTab==='portvar') buildPortVarChart();
  else if(currentRiskTab==='dts') buildDTSChart();
  else if(currentRiskTab==='dd') buildDrawdownChart();
}

function switchTab(tab,btn){
  currentTab=tab;
  // Only affect attribution panel tabs
  document.querySelectorAll('#attrPanel .tab').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('legCum').style.display=(tab==='cum'||tab==='weekly')?'flex':'none';
  document.getElementById('legOas').style.display=tab==='oas'?'flex':'none';
  document.getElementById('legCurve').style.display=tab==='curve'?'flex':'none';
  redraw();
}

function buildInfoPanel() {
  const s = DATA.series_sources;
  const rows = Object.entries(s).map(function(e) {
    const isLive = e[1].includes('FRED') || e[1].includes('yfinance');
    const color = e[1].includes('simulation') ? 'var(--carry)' : e[1].includes('yfinance') ? '#534AB7' : 'var(--pos)';
    return '<tr>'+
      '<td style="color:var(--t2);font-size:11px">'+e[0]+'</td>'+
      '<td style="font-size:11px;color:'+color+'">'+e[1]+'</td>'+
      '</tr>';
  }).join('');

  document.getElementById('infoPanel').innerHTML =
    '<table style="width:100%;border-collapse:collapse;font-size:12px;margin-bottom:10px">'+
    '<thead><tr>'+
    '<th style="font-size:11px;font-weight:500;color:var(--t2);text-align:left;padding:4px 8px;border-bottom:.5px solid var(--bdr)">Series</th>'+
    '<th style="font-size:11px;font-weight:500;color:var(--t2);text-align:left;padding:4px 8px;border-bottom:.5px solid var(--bdr)">Source</th>'+
    '</tr></thead><tbody>'+rows+'</tbody></table>'+
    '<div style="display:flex;flex-wrap:wrap;gap:16px;font-size:11px;color:var(--t3)">'+
    '<span>Period: '+DATA.data_start+' – '+DATA.data_end+'</span>'+
    '<span>Weekly periods: '+DATA.n_periods+'</span>'+
    '<span>Generated: '+DATA.generated+'</span>'+
    '<span>Framework: Campisi + KRD</span>'+
    '<span>DV01 notional: $10M</span>'+
    '</div>';
}

redraw();
buildInfoPanel();
</script>
</body>
</html>"""

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("Fixed Income Attribution Dashboard — v7 (KRD + PCA, UST yields)")
    print("="*60)
    print(f"\n[1/3] Fetching market data ({START_DATE} → {END_DATE})...")

    ig_oas = fetch_fred_csv("BAMLC0A0CM")
    hy_oas = fetch_fred_csv("BAMLH0A0HYM2")

    # Convert OAS % → bp
    if ig_oas is not None: ig_oas = ig_oas * 100
    if hy_oas is not None: hy_oas = hy_oas * 100
    have_oas = ig_oas is not None and hy_oas is not None

    # Yields: FRED's DGS series time out on some networks, so pull the real
    # 2/5/10/30 par curve straight from the US Treasury (no API key needed).
    print("\n  Fetching yields from US Treasury (par yield curve)...")
    ty = fetch_treasury_yields()
    have_rates = ty is not None

    if have_oas and have_rates:
        df = pd.DataFrame({"ig":ig_oas,"hy":hy_oas,
                           "r2":ty["r2"],"r5":ty["r5"],"r10":ty["r10"],"r30":ty["r30"]}
                          ).loc[START_DATE:END_DATE].dropna()
        market = {"lqd_oas":df["ig"],"hyg_oas":df["hy"],
                  "r2":df["r2"],"r5":df["r5"],"r10":df["r10"],"r30":df["r30"],
                  "source":"FRED OAS + US Treasury par yields"}
        series_sources = {
            "LQD OAS (BAMLC0A0CM)": "FRED", "HYG OAS (BAMLH0A0HYM2)": "FRED",
            "2y yield (UST 2 Yr)": "US Treasury", "5y yield (UST 5 Yr)": "US Treasury",
            "10y yield (UST 10 Yr)": "US Treasury", "30y yield (UST 30 Yr)": "US Treasury",
        }
        badge_class, data_label = "badge-live", "live · FRED + Treasury"
        data_source = "FRED — ICE BofA OAS + US Treasury par yields (2/5/10/30)"

    elif have_oas and not have_rates:
        print("\n  Treasury unavailable — falling back to yfinance proxies...")
        yf_yields = fetch_yfinance_yields()
        if yf_yields is not None:
            r2_use  = yf_yields.get("r2")
            r5_use  = yf_yields.get("r5")
            r30_use = yf_yields.get("r30")
            if any(x is None for x in [r2_use, r5_use, r30_use]):
                raise ValueError("Insufficient yield data from yfinance")
            df = pd.DataFrame({"ig":ig_oas,"hy":hy_oas,
                               "r2":r2_use,"r5":r5_use,"r30":r30_use}).dropna()
            market = {"lqd_oas":df["ig"],"hyg_oas":df["hy"],
                      "r2":df["r2"],"r5":df["r5"],"r30":df["r30"],
                      "source":"FRED OAS + yfinance yield proxies"}
            series_sources = {
                "LQD OAS (BAMLC0A0CM)": "FRED", "HYG OAS (BAMLH0A0HYM2)": "FRED",
                "2y yield": "yfinance ^IRX (3m proxy)", "5y yield": "yfinance ^FVX",
                "10y yield": "interpolated 5y→30y", "30y yield": "yfinance ^TYX",
            }
            badge_class, data_label = "badge-yf", "live · FRED + yf proxy"
            data_source = "FRED OAS + yfinance yield proxies (2y≈3m, 10y interpolated)"
        else:
            market = simulate_calibrated()
            series_sources = {"All series": "Calibrated simulation"}
            badge_class, data_label = "badge-sim", "calibrated simulation"
            data_source = "Calibrated simulation"

    else:
        market = simulate_calibrated()
        series_sources = {"All series": "Calibrated simulation (anchored to real levels)"}
        badge_class, data_label = "badge-sim", "calibrated simulation"
        data_source = "Calibrated simulation (anchored to real levels)"

    # Ensure a 10y node exists for the PCA module (interpolate if a branch lacked it)
    if market.get("r10") is None:
        market["r10"] = market["r5"] + 0.2*(market["r30"] - market["r5"])

    print("\n[2/3] Computing weekly attribution...")
    def bw(s): return build_weekly(s)

    lqd_oas_w = bw(market["lqd_oas"])
    hyg_oas_w = bw(market["hyg_oas"])
    r2_w  = bw(market["r2"])
    r5_w  = bw(market["r5"])
    r10_w = bw(market["r10"])
    r30_w = bw(market["r30"])

    common_w = (lqd_oas_w.index.intersection(hyg_oas_w.index)
                .intersection(r2_w.index).intersection(r5_w.index)
                .intersection(r10_w.index).intersection(r30_w.index))

    lqd_oas_w = lqd_oas_w.loc[common_w]
    hyg_oas_w = hyg_oas_w.loc[common_w]
    rates_w = {"r2": r2_w.loc[common_w], "r5": r5_w.loc[common_w],
               "r10": r10_w.loc[common_w], "r30": r30_w.loc[common_w]}

    # Diagnostic
    print(f"  LQD OAS: {lqd_oas_w.min():.0f}–{lqd_oas_w.max():.0f} bp")
    print(f"  HYG OAS: {hyg_oas_w.min():.0f}–{hyg_oas_w.max():.0f} bp")
    print(f"  2y: {rates_w['r2'].min():.2f}–{rates_w['r2'].max():.2f}%")
    print(f"  5y: {rates_w['r5'].min():.2f}–{rates_w['r5'].max():.2f}%")
    print(f"  10y:{rates_w['r10'].min():.2f}–{rates_w['r10'].max():.2f}%")
    print(f"  30y:{rates_w['r30'].min():.2f}–{rates_w['r30'].max():.2f}%")

    lqd_attr = compute_attribution(lqd_oas_w, rates_w, **ETF_PARAMS["LQD"])
    hyg_attr = compute_attribution(hyg_oas_w, rates_w, **ETF_PARAMS["HYG"])

    common_a = lqd_attr.index.intersection(hyg_attr.index)
    lqd_attr = lqd_attr.loc[common_a]
    hyg_attr = hyg_attr.loc[common_a]

    # ── Yield-curve PCA decomposition of the rate component ──────────────────────
    pca   = compute_pca(rates_w, common_a)
    k_lqd = krd_grid(ETF_PARAMS["LQD"])
    k_hyg = krd_grid(ETF_PARAMS["HYG"])

    def add_factor_contrib(attr_df, k_vec):
        # factor duration β_f = KRD · v_f ; contribution_f,t = −β_f × score_f,t
        betas = {f: float(np.dot(k_vec, pca["vec"][f])) for f in ["level","slope","curv","resid"]}
        sc = pca["scores"].reindex(attr_df.index)
        for f in ["level","slope","curv","resid"]:
            attr_df["rate_"+f] = (-betas[f] * sc[f]).fillna(0.0)
        return betas

    beta_lqd = add_factor_contrib(lqd_attr, k_lqd)
    beta_hyg = add_factor_contrib(hyg_attr, k_hyg)

    print("\n  PCA variance explained: " +
          "  ".join(f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(pca["var"])))
    for nm in ["level","slope","curv"]:
        print(f"    {nm:5s} @[2,5,10,30]: " + " ".join(f"{x:+.2f}" for x in pca["vec"][nm]))
    print(f"  Factor duration LQD  L/S/C: {beta_lqd['level']:.2f} / {beta_lqd['slope']:.2f} / {beta_lqd['curv']:.2f}")
    print(f"  Factor duration HYG  L/S/C: {beta_hyg['level']:.2f} / {beta_hyg['slope']:.2f} / {beta_hyg['curv']:.2f}")
    # Reconciliation check: Level+Slope+Curv+Resid should equal rate total
    _chk = (lqd_attr[["rate_level","rate_slope","rate_curv","rate_resid"]].sum(axis=1) - lqd_attr["rate"]).abs().max()
    print(f"  LQD reconciliation (max |L+S+C+R − rate|): {_chk:.2e}%")

    print(f"  {len(lqd_attr)} weekly periods")
    print(f"  LQD: total={lqd_attr['total'].sum():.2f}%  "
          f"rate_near={lqd_attr['rate_near'].sum():.2f}%  rate_far={lqd_attr['rate_far'].sum():.2f}%")
    print(f"  HYG: total={hyg_attr['total'].sum():.2f}%  "
          f"rate_near={hyg_attr['rate_near'].sum():.2f}%  rate_far={hyg_attr['rate_far'].sum():.2f}%")

    # ── Risk metrics ────────────────────────────────────────────────────────────
    # Using last observation for point-in-time metrics
    lqd_oas_now = float(lqd_oas_w.iloc[-1])
    hyg_oas_now = float(hyg_oas_w.iloc[-1])
    lqd_dur     = ETF_PARAMS["LQD"]["duration"]
    hyg_dur     = ETF_PARAMS["HYG"]["duration"]

    # DTS = Duration × OAS (bp)
    lqd_dts = lqd_dur * lqd_oas_now
    hyg_dts = hyg_dur * hyg_oas_now

    # DV01 per $10M notional = Duration × 0.0001 × $10M
    NOTIONAL = 10_000_000
    lqd_dv01 = lqd_dur * 0.0001 * NOTIONAL
    hyg_dv01 = hyg_dur * 0.0001 * NOTIONAL

    # KRD DV01 per $10M — split by node
    lqd_dv01_5  = ETF_PARAMS["LQD"]["krd_near"] * 0.0001 * NOTIONAL
    lqd_dv01_30 = ETF_PARAMS["LQD"]["krd_far"]  * 0.0001 * NOTIONAL
    hyg_dv01_2  = ETF_PARAMS["HYG"]["krd_near"] * 0.0001 * NOTIONAL
    hyg_dv01_5  = ETF_PARAMS["HYG"]["krd_far"]  * 0.0001 * NOTIONAL

    # Spread P&L series for VaR
    doas_lqd = lqd_oas_w.diff().dropna()
    doas_hyg = hyg_oas_w.diff().dropna()

    def spread_pnl_series(doas_bp, duration):
        return (-duration * doas_bp / 10000 * 100).values  # % return array

    lqd_pnl = spread_pnl_series(doas_lqd, lqd_dur)
    hyg_pnl = spread_pnl_series(doas_hyg, hyg_dur)

    # Align lengths
    min_len = min(len(lqd_pnl), len(hyg_pnl))
    lqd_pnl = lqd_pnl[-min_len:]
    hyg_pnl = hyg_pnl[-min_len:]

    # Correlation between LQD and HYG spread P&L
    corr_lqd_hyg = float(np.corrcoef(lqd_pnl, hyg_pnl)[0, 1])

    def compute_var(pnl_array, conf=0.95, holding_days=10):
        pct = np.percentile(pnl_array, (1-conf)*100)
        scale = np.sqrt(holding_days / 5)
        var  = -pct  * scale
        cvar = -pnl_array[pnl_array <= pct].mean() * scale
        return round(float(var), 3), round(float(cvar), 3)

    lqd_svar, lqd_scvar = compute_var(lqd_pnl)
    hyg_svar, hyg_scvar = compute_var(hyg_pnl)

    def portfolio_var(w_lqd, w_hyg, s1, s2, rho, holding_days=10):
        """σ_p = √(w1²σ1² + w2²σ2² + 2·w1·w2·σ1·σ2·ρ), scaled to 10-day."""
        var_p = np.sqrt(w_lqd**2 * s1**2 + w_hyg**2 * s2**2 +
                        2 * w_lqd * w_hyg * s1 * s2 * rho)
        return round(float(var_p), 3)

    # 1-day VaR σ (not scaled) for correlation formula
    lqd_var1d = float(-np.percentile(lqd_pnl, 5))
    hyg_var1d = float(-np.percentile(hyg_pnl, 5))

    # VaR vs weight curve (0% to 100% LQD in 5% steps)
    var_curve_x   = list(range(0, 101, 5))
    var_curve_y   = [portfolio_var(w/100, 1-w/100, lqd_var1d, hyg_var1d, corr_lqd_hyg)
                     * np.sqrt(10/5) for w in var_curve_x]
    var_naive_y   = [(w/100 * lqd_var1d + (1-w/100) * hyg_var1d) * np.sqrt(10/5)
                     for w in var_curve_x]  # no diversification

    port_var_60_40 = portfolio_var(0.60, 0.40, lqd_var1d, hyg_var1d, corr_lqd_hyg) * np.sqrt(10/5)
    divers_benefit = round(float((0.60*lqd_svar + 0.40*hyg_svar) - port_var_60_40), 3)

    # Rolling DTS (weekly)
    rolling_lqd_dts = (lqd_oas_w * lqd_dur).tolist()
    rolling_hyg_dts = (hyg_oas_w * hyg_dur).tolist()

    # IG vs HY risk contribution ratio (DTS-weighted)
    def risk_contrib(w_lqd, w_hyg, dts_lqd, dts_hyg):
        total = w_lqd * dts_lqd + w_hyg * dts_hyg
        if total == 0: return 0.5, 0.5
        return round(w_lqd * dts_lqd / total, 3), round(w_hyg * dts_hyg / total, 3)

    rc_lqd_60, rc_hyg_60 = risk_contrib(0.60, 0.40, lqd_dts, hyg_dts)

    # Max drawdown on weekly total return
    def max_drawdown(ret_series):
        cum = (1 + ret_series/100).cumprod()
        roll_max = cum.cummax()
        dd = (cum - roll_max) / roll_max * 100
        return round(float(dd.min()), 2), dd.tolist()

    lqd_mdd, lqd_dd_series = max_drawdown(lqd_attr["total"])
    hyg_mdd, hyg_dd_series = max_drawdown(hyg_attr["total"])

    risk_metrics = {
        "lqd": {
            "dts":      round(lqd_dts, 1),
            "dv01":     round(lqd_dv01, 0),
            "dv01_5":   round(lqd_dv01_5, 0),
            "dv01_30":  round(lqd_dv01_30, 0),
            "svar_10d": lqd_svar,
            "scvar_10d":lqd_scvar,
            "mdd":      lqd_mdd,
        },
        "hyg": {
            "dts":      round(hyg_dts, 1),
            "dv01":     round(hyg_dv01, 0),
            "dv01_2":   round(hyg_dv01_2, 0),
            "dv01_5":   round(hyg_dv01_5, 0),
            "svar_10d": hyg_svar,
            "scvar_10d":hyg_scvar,
            "mdd":      hyg_mdd,
        },
        "corr":            round(corr_lqd_hyg, 3),
        "port_var_60_40":  round(port_var_60_40, 3),
        "divers_benefit":  divers_benefit,
        "var_curve_x":     var_curve_x,
        "var_curve_y":     [round(v, 3) for v in var_curve_y],
        "var_naive_y":     [round(v, 3) for v in var_naive_y],
        "rolling_lqd_dts": [round(x,1) for x in rolling_lqd_dts],
        "rolling_hyg_dts": [round(x,1) for x in rolling_hyg_dts],
        "lqd_dd":  [round(x,3) for x in lqd_dd_series],
        "hyg_dd":  [round(x,3) for x in hyg_dd_series],
        "rc_lqd_60": rc_lqd_60,
        "rc_hyg_60": rc_hyg_60,
        "notional": "10M",
    }

    print(f"  LQD DTS={lqd_dts:.0f}  DV01=${lqd_dv01:,.0f}  SpreadVaR(10d)={lqd_svar:.2f}%  MDD={lqd_mdd:.2f}%")
    print(f"  HYG DTS={hyg_dts:.0f}  DV01=${hyg_dv01:,.0f}  SpreadVaR(10d)={hyg_svar:.2f}%  MDD={hyg_mdd:.2f}%")
    print(f"  Correlation LQD-HYG spread: {corr_lqd_hyg:.3f}")
    print(f"  Portfolio VaR (60/40, 10d): {port_var_60_40:.2f}%  Diversification benefit: {divers_benefit:.2f}%")

    print("\n[3/3] Building HTML...")
    chart_labels = [d.strftime("%b %d") for d in lqd_attr.index]
    oas_dates    = [d.strftime("%Y-%m-%d") for d in lqd_oas_w.loc[common_a].index]

    def to_list(df):
        return [{"spread":    round(float(r.spread),     4),
                 "carry":     round(float(r.carry),      4),
                 "rate":      round(float(r.rate),       4),
                 "rate_near": round(float(r.rate_near),  4),
                 "rate_far":  round(float(r.rate_far),   4),
                 "rate_level":round(float(r.rate_level), 4),
                 "rate_slope":round(float(r.rate_slope), 4),
                 "rate_curv": round(float(r.rate_curv),  4),
                 "rate_resid":round(float(r.rate_resid), 4),
                 "convexity": round(float(r.convexity),  4),
                 "total":     round(float(r.total),      4)}
                for _, r in df.iterrows()]

    data_json = json.dumps({
        "labels":        chart_labels,
        "oas_dates":     oas_dates,
        "lqd_oas":       [round(float(x),2) for x in lqd_oas_w.loc[common_a]],
        "hyg_oas":       [round(float(x),2) for x in hyg_oas_w.loc[common_a]],
        "r2":            [round(float(x),3) for x in rates_w["r2"].loc[common_a]],
        "r5":            [round(float(x),3) for x in rates_w["r5"].loc[common_a]],
        "r10":           [round(float(x),3) for x in rates_w["r10"].loc[common_a]],
        "r30":           [round(float(x),3) for x in rates_w["r30"].loc[common_a]],
        "lqd":           to_list(lqd_attr),
        "hyg":           to_list(hyg_attr),
        "risk":          risk_metrics,
        "source":        market["source"],
        "series_sources": series_sources,
        "data_start":    oas_dates[0] if oas_dates else START_DATE,
        "data_end":      oas_dates[-1] if oas_dates else END_DATE,
        "generated":     datetime.today().strftime("%Y-%m-%d %H:%M"),
        "n_periods":     len(lqd_attr),
        "pca": {
            "tenors":    pca["tenors"],
            "vec_level": [round(float(x),4) for x in pca["vec"]["level"]],
            "vec_slope": [round(float(x),4) for x in pca["vec"]["slope"]],
            "vec_curv":  [round(float(x),4) for x in pca["vec"]["curv"]],
            "var":       [round(float(x),4) for x in pca["var"]],
            "beta_lqd":  {k: round(float(v),3) for k,v in beta_lqd.items()},
            "beta_hyg":  {k: round(float(v),3) for k,v in beta_hyg.items()},
        },
    })

    html = HTML
    html = html.replace("__BADGE_CLASS__",  badge_class)
    html = html.replace("__DATA_LABEL__",   data_label)
    html = html.replace("__START_DATE__",   START_DATE)
    html = html.replace("__END_DATE__",     END_DATE)
    html = html.replace("__DATA_SOURCE__",  data_source)
    html = html.replace("__GENERATED__",    datetime.today().strftime("%Y-%m-%d"))
    html = html.replace("__DATA_JSON__",    data_json)
    html = html.replace("__LQD_KRD_5__",   f"{LQD_KRD_5:.2f}")
    html = html.replace("__LQD_KRD_30__",  f"{LQD_KRD_30:.2f}")
    html = html.replace("__HYG_KRD_2__",   f"{HYG_KRD_2:.2f}")
    html = html.replace("__HYG_KRD_5__",   f"{HYG_KRD_5:.2f}")

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_FILE)), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✓  Saved → {OUTPUT_FILE}")
    print("   Open in any browser — no server needed.\n")

if __name__ == "__main__":
    main()

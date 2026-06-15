# Fixed Income Portfolio Attribution

An interactive fixed income attribution dashboard built on the **Campisi framework**,
**Key Rate Duration (KRD)**, and a **yield-curve PCA** module that splits the rate
return into orthogonal Level / Slope / Curvature factors.

Single self-contained HTML file — no server required.

![Dashboard preview](dashboard_preview.png)

## What it does

Decomposes weekly portfolio return for two credit ETFs (LQD = IG, HYG = HY) into:

- **Rate** — price impact of Treasury curve moves, via Key Rate Duration
  - LQD: 5y + 30y nodes · HYG: 2y + 5y nodes
  - further split into **Level / Slope / Curvature** via yield-curve PCA
- **Spread** — price impact of OAS changes (−Duration × ΔOAS)
- **Carry** — coupon income (coupon ÷ 52)
- **Convexity** — second-order correction for large rate/spread moves
  (the bond's own price–yield convexity — *distinct* from curve curvature)

Portfolio weights are adjustable in real time. The dashboard also reports DTS, DV01,
Spread VaR/CVaR with correlation and diversification benefit, rolling DTS, and drawdown.

## Yield-curve PCA module

PCA on weekly changes of the 2/5/10/30 curve:

```
covariance matrix  →  eigendecomposition  →  Level / Slope / Curvature
```

- **Eigenvectors** are the factor shapes (Level ≈ flat, Slope ≈ tilt, Curvature ≈ hump);
  the first three explain 95%+ of curve variance.
- **Factor duration** β_f = KRD · v_f maps each ETF's KRD onto a factor.
- **Contribution** of factor f = −β_f × (factor score), where the score is the curve
  move projected onto the eigenvector.
- Because the eigenvectors are orthogonal, the Level/Slope/Curvature split is additive
  and **reconciles with the KRD rate total to machine precision** — it refines the rate
  component rather than replacing anything.

The panel shows the factor shapes, variance explained, per-ETF factor durations, and a
weight-aware cumulative L/S/C rate split.

## Data sources

| Series | Source |
|---|---|
| IG OAS (BAMLC0A0CM), HY OAS (BAMLH0A0HYM2) | FRED |
| 2y / 5y / 10y / 30y yields | US Treasury — Daily Par Yield Curve |

Fallback chain for yields: **US Treasury → yfinance proxies → calibrated simulation**.
(FRED's DGS constant-maturity series are not used — they time out on some networks;
the US Treasury par curve is a no-key, reliable substitute with a real 2y and 10y.)

## Usage

```bash
pip install pandas numpy requests yfinance
python fetch_and_build_v7.py
```

Generates a self-contained HTML file (data baked in, Plotly loaded from CDN). Open in any
browser — no server needed.

## Stack

Python · Plotly.js · FRED · US Treasury · yfinance (fallback)

## Notes

- The HTML loads Plotly from a CDN, so viewing it requires an internet connection.
- Curve curvature (PC3) typically explains only ~1–3% of variance; over short samples its
  shape is the least stable of the three factors. A longer history (earlier `START_DATE`)
  stabilises it.

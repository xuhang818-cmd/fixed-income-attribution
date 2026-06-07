\# Fixed Income Portfolio Attribution



An interactive fixed income portfolio attribution dashboard built with the Campisi framework and Key Rate Duration (KRD).



\## What it does



Decomposes portfolio returns into four components:



\- \*\*Rate contribution\*\* — broken down by key rate nodes using KRD interpolation

&#x20; - LQD (IG): 5y and 30y nodes (DGS5 + DGS30)

&#x20; - HYG (HY): 2y and 5y nodes (DGS2 + ^IRX + DGS5)

\- \*\*Spread contribution\*\* — price impact from OAS changes (−Duration × ΔOAS)

\- \*\*Carry\*\* — weekly coupon income

\- \*\*Convexity\*\* — second-order correction for large rate/spread moves



Portfolio weights are adjustable in real time via sliders. Four chart views: cumulative attribution, weekly stacked bars, OAS levels, and yield curve (2y/5y/30y).



\## Usage



```bash

pip install pandas numpy requests yfinance

python fetch\_and\_build\_v4.py

```



Generates a self-contained HTML file. Open in any browser — no server needed. Data from FRED (OAS indices + DGS2/5/30) with automatic fallback to yfinance and calibrated simulation.



\## Stack



Python · Plotly.js · FRED API · yfinance


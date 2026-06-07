# \# Fixed Income Portfolio Attribution

# 

# A fixed income portfolio attribution demo built with the Campisi framework.

# 

# \## What it does

# 

# Decomposes weekly portfolio returns into four components:

# \- \*\*Rate contribution\*\* — price impact from changes in the 10y Treasury yield

# \- \*\*Spread contribution\*\* — price impact from changes in OAS

# \- \*\*Carry\*\* — coupon income

# \- \*\*Convexity\*\* — second-order correction for large rate/spread moves

# 

# \## Data

# 

# Live data from FRED (ICE BofA OAS indices) and yfinance (^TNX), with automatic fallback to calibrated simulation if unavailable.

# 

# Proxies used: LQD (investment grade) and HYG (high yield).

# 

# \## Usage

# 

# ```bash

# pip install pandas numpy requests yfinance

# python fetch\_and\_build\_v3.py

# ```

# 

# Generates a self-contained HTML dashboard. Open in any browser — no server needed.

# 

# \## Stack

# 

# Python · Plotly.js · FRED API · yfinance


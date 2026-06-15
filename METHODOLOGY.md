# Methodology — Yield-Curve PCA

The PCA module splits the rate component of the Campisi/KRD attribution into level / slope / curvature.

## 1. Purpose

The base attribution builds the rate component from KRDs at two nodes per ETF (LQD: 5y, 30y; HYG: 2y, 5y). Two nodes capture level and slope but not curvature: two points define a line, so they can't see the belly move against the wings. This module adds curvature and re-expresses the rate move in orthogonal factors.

Two different "curvature" concepts are kept separate:

- Convexity (Campisi): a bond's second-order price sensitivity to its own yield, an instrument property. Untouched here.
- Curvature (PCA): the shape of the curve's move across maturities (butterfly). What this adds.

This refines the rate component; it is not a substitute for the convexity term.

## 2. Method

Let Δy_t be the weekly change vector at [2,5,10,30]. Centre the columns, form the 4×4 covariance Σ, and take its eigendecomposition Σ = V Λ Vᵀ (descending eigenvalues, orthonormal eigenvectors).

- eigenvector v_i is the factor shape (a weight per maturity)
- eigenvalue λ_i is the variance along it; variance explained = λ_i / Σλ

For the Treasury curve, PC1 is level (~85%), PC2 slope (~5–10%), PC3 curvature (~1–3%); the three explain 95%+ (Litterman & Scheinkman, 1991). The eigenvectors pick up one sign change each (level 0, slope 1, curvature 2). Signs are arbitrary and fixed to a convention in code: level positive, slope increasing, curvature wings-positive.

The factor score each week is the curve move projected onto the shape:

    score_{f,t} = v_f · Δy_t

Scores use the raw (uncentred) change, since the eigenvectors are a complete orthonormal basis and any move decomposes exactly.

## 3. KRD ↔ PCA

Rate return is r_rate ≈ −KRD · Δy. Place each ETF's two-node KRD on the [2,5,10,30] grid (zero off-node) and project onto the eigenvectors:

    β_f = KRD · v_f                  (factor duration)
    contribution_f = −β_f × score_f

Substituting the factor reconstruction of Δy:

    r_rate ≈ −KRD · Δy = −Σ_f (KRD · v_f) score_f = −Σ_f β_f score_f

so level + slope + curvature + residual = rate total. In code this reconciles to ~1e-15.

Raw KRD buckets are correlated (they share the common level move), so a bucket-by-bucket split double-counts. The PCA factors are orthogonal, so the contributions are additive.

## 4. Implementation

- Yields: US Treasury daily par curve (2/5/10/30), no key. DGS is not used (timeouts). Fallback: yfinance → simulation.
- OAS: FRED ICE BofA (BAMLC0A0CM, BAMLH0A0HYM2).
- PCA: weekly changes, covariance matrix (not correlation), centred; symmetric eigensolver, descending eigenvalues; slope/curvature order checked by sign-change count.
- Output: per-ETF rate_level / rate_slope / rate_curv / rate_resid, factor durations, eigenvector shapes.

Checks: the eigenvectors should look like a flat line, a tilt, and a hump; distorted shapes mean a data problem. A near-zero PC4 signals a non-independent tenor (e.g. an interpolated 10y makes Σ rank-deficient, giving a zero eigenvalue); real data gives a small non-zero PC4.

Caveat: curvature carries little variance, so over short samples PC3's shape and the curvature durations are the least stable of the three, and which node carries curvature depends on where the belly sits in the sample. A longer history helps.

## Reference

Litterman, R. and Scheinkman, J. (1991). Common Factors Affecting Bond Returns. Journal of Fixed Income, 1(1), 54–61.

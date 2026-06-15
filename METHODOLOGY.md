# Methodology — Yield-Curve PCA for Fixed Income Attribution

Technical documentation for the PCA module that splits the rate component of the
Campisi/KRD attribution into orthogonal Level / Slope / Curvature factors.

---

## 1. Why this module exists

The base attribution decomposes return into rate / spread / carry / convexity, with the
rate component built from Key Rate Durations at two nodes per ETF (LQD: 5y, 30y;
HYG: 2y, 5y). Two nodes implicitly capture parallel (Level) and steepening/flattening
(Slope) moves, but two points define a straight line — they cannot represent **curvature**
(belly vs. wings). This module adds the missing curvature dimension and re-expresses the
rate move in orthogonal factors.

Note the two distinct "curvature" concepts:

- **Convexity** (Campisi term) — a bond's own second-order price sensitivity to *its own*
  yield. An instrument property. Left untouched by this module.
- **Curvature** (PCA factor) — the *shape* of the curve's movement across maturities
  (butterfly). What this module adds.

They are not substitutes; this module refines the **rate** component, not the convexity term.

---

## 2. Intuition

Each day, the yields at 2/5/10/30 change by some amount — a point in 4-D space. Across a
sample, these points form a thin, stretched cloud, because curve moves are highly
correlated (when the 2y rises the 10y usually rises too). PCA finds the principal axes of
that cloud. Although the data live in 4 dimensions, ~3 shapes describe almost everything:

- **Level** — all tenors move together (≈ 85% of variance)
- **Slope** — short and long ends move oppositely (≈ 5–10%)
- **Curvature** — belly moves against the wings (≈ 1–3%)

Together ~95%+. (Litterman & Scheinkman, 1991.)

---

## 3. The mathematics

### 3.1 Covariance and eigendecomposition

Let Δy_t be the weekly change vector at [2,5,10,30]. Centre the columns and form the
covariance matrix Σ (4×4, symmetric, positive semi-definite). Its eigendecomposition

    Σ = V Λ Vᵀ ,   eigenvalues λ₁ ≥ λ₂ ≥ … ,   orthonormal eigenvectors v₁, v₂, …

gives:

- **eigenvector vᵢ** = the *shape* of factor i (a weight per maturity)
- **eigenvalue λᵢ** = the *variance* along that shape; variance explained = λᵢ / Σλ

Symmetry guarantees the eigenvectors are orthogonal — this is what makes the resulting
factors uncorrelated, and the attribution clean.

### 3.2 Why the eigenvectors are Level / Slope / Curvature

Because tenor correlations decay smoothly with maturity distance, the eigenvectors behave
like the harmonics of a string: each successive one adds a sign change. PC1 has zero sign
changes (Level), PC2 has one (Slope), PC3 has two (Curvature). Eigenvector signs are
arbitrary and are fixed to a readable convention in code (Level positive, Slope increasing,
Curvature wings-positive).

### 3.3 Factor scores

The amount of each factor on a given week is the curve move **projected** onto the shape:

    score_{f,t} = v_f · Δy_t        (a dot product)

Scores are taken on the **raw** (uncentred) weekly change, because the eigenvectors form a
complete orthonormal basis of R⁴ — any move decomposes exactly.

---

## 4. The KRD ↔ PCA bridge

The rate component is r_rate ≈ −KRD · Δy. Each ETF's two-node KRD is placed on the common
[2,5,10,30] grid (zero where it has no node), then projected onto the eigenvectors:

    factor duration   β_f = KRD · v_f
    contribution_f    = −β_f × score_f

Substituting the factor reconstruction of Δy shows the split is exact:

    r_rate ≈ −KRD · Δy = −Σ_f (KRD · v_f) score_f = −Σ_f β_f score_f

So Level + Slope + Curvature + Residual = rate total. In the implementation this reconciles
to machine precision (max |L+S+C+R − rate| ≈ 1e-15 %).

**Why this beats raw KRD-bucket attribution:** raw buckets are correlated (they share the
common Level move), so a bucket-by-bucket split double-counts. PCA factors are orthogonal,
so the contributions are non-overlapping and additive.

---

## 5. Implementation

- **Yields:** US Treasury Daily Par Yield Curve (2/5/10/30), no API key. FRED's DGS series
  are not used (they time out on some networks). Fallback: yfinance proxies → simulation.
- **OAS:** FRED ICE BofA indices (BAMLC0A0CM, BAMLH0A0HYM2).
- **PCA:** weekly changes, covariance matrix (not correlation), centred for estimation;
  symmetric eigensolver, eigenvalues sorted descending; slope/curvature order guarded by
  sign-change count.
- **Output:** per-ETF rate_level / rate_slope / rate_curv / rate_resid columns, plus factor
  durations and eigenvector shapes, surfaced in a dedicated PCA panel.

### Sanity checks

- Plot the eigenvectors: Level ≈ flat, Slope ≈ monotonic tilt, Curvature ≈ single hump.
  Distorted shapes indicate a data problem (e.g. a proxy or interpolated tenor).
- A near-zero PC4 is a fingerprint of a non-independent tenor (e.g. an interpolated 10y
  makes the covariance rank-deficient → a zero eigenvalue). Real data gives a small,
  non-zero PC4.

### Known limitations

- Curvature (PC3) carries little variance; over short samples its shape — and therefore the
  curvature factor durations — are the least stable of the three. A longer history
  stabilises it. Which node carries curvature depends on where the empirical belly sits in
  the sample period, so curvature exposure is sample-dependent.

---

## Reference

Litterman, R. and Scheinkman, J. (1991). *Common Factors Affecting Bond Returns.*
Journal of Fixed Income, 1(1), 54–61.

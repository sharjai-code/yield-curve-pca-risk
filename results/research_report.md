# Reproducible research results

Data: 2025-01-02 to 2025-12-31; 249 complete curves and 248 daily changes.

The first three covariance PCs explain **98.68%** of centered raw-bp variance. The correlation PCA result is **98.55%** of standardized variance; the denominators differ.

The circular block-bootstrap 95% percentile interval for the covariance top-three share is **[98.44%, 99.23%]** using 500 replications, blocks of 10 observations, and seed 42.

Holdout models were fitted through **2025-10-17**. The holdout starts **2025-10-20** and contains 50 observed shocks. This evaluates reconstruction of observed changes, not forecasting.

## Reconstruction

| Mode | PCs | Sample | RMSE (bp) | Raw-bp energy captured |
|---|---:|---|---:|---:|
| covariance | 1 | full_sample | 2.053 | 82.37% |
| covariance | 1 | holdout | 1.410 | 81.39% |
| covariance | 2 | full_sample | 0.819 | 97.19% |
| covariance | 2 | holdout | 0.743 | 94.83% |
| covariance | 3 | full_sample | 0.563 | 98.68% |
| covariance | 3 | holdout | 0.401 | 98.50% |
| correlation | 1 | full_sample | 2.058 | 82.28% |
| correlation | 1 | holdout | 1.414 | 81.28% |
| correlation | 2 | full_sample | 0.832 | 97.10% |
| correlation | 2 | holdout | 0.755 | 94.67% |
| correlation | 3 | full_sample | 0.588 | 98.55% |
| correlation | 3 | holdout | 0.515 | 97.52% |

## Portfolio risk

Illustrative regular-coupon bonds: $1m face each at 2, 5, 10, and 30 years; 4% coupons; semiannual payments; no accrued interest.

Portfolio value: **$3,858,203.04**. Parallel DV01: **$2,857.40/bp**.

| Scenario | Full repricing P&L ($) | Linear P&L ($) | Nonlinear residual ($) |
|---|---:|---:|---:|
| parallel_+100bp | -266,036.75 | -285,740.08 | 19,703.33 |
| parallel_-100bp | 308,549.77 | 285,740.08 | 22,809.69 |
| steepener_-25_to_+25bp | -27,905.87 | -28,779.72 | 873.85 |
| PC1_+1sigma | -12,483.56 | -12,516.96 | 33.40 |
| PC1_-1sigma | 12,550.50 | 12,516.96 | 33.55 |
| PC2_+1sigma | -3,344.68 | -3,358.95 | 14.27 |
| PC2_-1sigma | 3,373.29 | 3,358.95 | 14.34 |
| PC3_+1sigma | -459.06 | -460.60 | 1.54 |
| PC3_-1sigma | 462.14 | 460.60 | 1.54 |

## Interpretation and limits

- Level, slope, and curvature are economic interpretations; a given sample need not recover textbook shapes or a fixed ordering.
- Covariance PCA prioritizes maturities with more raw-bp volatility; correlation PCA equalizes training-sample maturity volatility.
- Trailing PCA is a descriptive diagnostic. A fixed initial-window reference is used for cosine matching and subspace angles.
- Block bootstrap preserves dependence within blocks, not long-run nonstationarity. Loading intervals are pointwise and become fragile near tied eigenvalues.
- Bond cashflows are discounted with continuously compounded zero yields and linear interpolation of log discount factors. Cashflows below one year use the origin-to-one-year segment.
- Historical holdout shocks are applied to a fixed latest-date curve and fixed cashflows. Their P&L is a static stress study, not realized returns or a trading backtest.
- These Treasury curves model interest-rate risk, not corporate-credit spreads, defaults, or liquidity risk.

## Provenance

Data: Federal Reserve Board, Gürkaynak–Sack–Wright nominal yield curve. Current vintages may revise historical observations.
Inspiration: Andrea Ranzato, pca-yield-curve-modelling. This project is an independently coded methodological adaptation, not a numerical reproduction of the dissertation.
Input SHA-256: `bfbb1bcaca6440e2ee503d704f03de49ad87a17c3da16fe0416f78d5e06e7620`.

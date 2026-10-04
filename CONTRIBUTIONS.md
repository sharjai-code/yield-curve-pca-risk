# Project provenance and contributions

Maintainer: Somya Harjai / [sharjai-code](https://github.com/sharjai-code).

The initial implementation was developed with ChatGPT assistance. The repository is a transparent adaptation of publicly documented yield-curve PCA methods, inspired by Andrea Ranzato's dissertation repository. Maintaining this project does not imply authorship of that dissertation or its upstream code.

## Included implementation

- Independently written Python modules for PCA, block bootstrap, rolling diagnostics, and bond risk.
- A frozen 2025 Federal Reserve GSW zero-coupon dataset with source and content hash.
- Covariance-versus-correlation comparisons using raw-bp reconstruction errors.
- Chronological holdout evaluation with frozen training transformations.
- Bootstrap factor matching and subspace diagnostics for rolling stability.
- Coupon-bond discounting, key-rate DV01, and full-versus-linear scenario repricing.
- A research notebook, generated report, numerical checks, and CI configuration.

## How to describe the work accurately

After reviewing and understanding the implementation, a suitable project description is:

> Reproduced and extended yield-curve PCA methods in Python using public Treasury zero-coupon data; evaluated factor stability, chronological holdout reconstruction, and bond portfolio sensitivities under yield-curve stress scenarios.

This description concerns an independently implemented adaptation. The project does not claim reproduction of the dissertation's exact numerical results, forecast skill, profitable trading results, or corporate-credit modeling.

## Next research steps

1. Run the same workflow on a longer frozen GSW vintage and compare market regimes.
2. Evaluate bootstrap block lengths of 5, 10, and 20 observations.
3. Investigate factor interpretation and near-degenerate eigenvalues before assigning economic labels.
4. Compare interpolation and valuation against a market-instrument library.
5. Add factor-neutral hedging with explicit funding, instrument, and position constraints.

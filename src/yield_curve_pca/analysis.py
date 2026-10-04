"""Command-line research pipeline; all tables and figures are reproducible."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from .curves import Bond, key_rate_dv01, price_portfolio
from .data import load_yields, yield_changes_bp, download_gsw
from .pca import fit_pca, block_bootstrap, rolling_pca


def run_analysis(data_path: str | Path, output_dir: str | Path = "results",
                 start: str | None = None, end: str | None = None,
                 holdout_fraction: float = .2, window: int = 63,
                 bootstrap_repetitions: int = 500, block_length: int = 10,
                 seed: int = 42) -> dict:
    """Run descriptive PCA and chronological holdout reconstruction diagnostics."""
    if not .05 <= holdout_fraction <= .5:
        raise ValueError("Holdout fraction must lie between 0.05 and 0.5.")
    yields = load_yields(data_path, start=start, end=end)
    changes = yield_changes_bp(yields)
    cutoff = int(len(changes) * (1 - holdout_fraction))
    train, test = changes.iloc[:cutoff], changes.iloc[cutoff:]
    if len(train) <= len(changes.columns):
        raise ValueError("Not enough training observations.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mats = yields.columns.to_numpy(dtype=float)
    models = {mode: fit_pca(changes, mode) for mode in ("covariance", "correlation")}
    trained = {mode: fit_pca(train, mode) for mode in models}
    retained = 3
    variance_rows = []
    reconstruction_rows = []
    for mode, model in models.items():
        for j in range(len(mats)):
            variance_rows.append({"mode": mode, "factor": f"PC{j+1}",
                                  "eigenvalue": model.eigenvalues[j],
                                  "variance_ratio": model.explained_variance_ratio[j],
                                  "cumulative_variance": model.explained_variance_ratio[:j+1].sum()})
        for k in (1, 2, 3):
            for split, values, fitted in (("full_sample", changes, model), ("holdout", test, trained[mode])):
                errors = values.to_numpy() - fitted.reconstruct(values.to_numpy(), k)
                baseline_errors = values.to_numpy() - fitted.mean
                reconstruction_rows.append({"mode": mode, "components": k, "sample": split,
                                            "rmse_bp": np.sqrt(np.mean(errors**2)),
                                            "raw_bp_energy_captured": 1-np.sum(errors**2)/np.sum(baseline_errors**2)})
        pd.DataFrame(model.components[:retained].T, index=mats,
                     columns=["PC1", "PC2", "PC3"]).rename_axis("maturity_years").to_csv(output/f"{mode}_loadings.csv")
        pd.DataFrame(model.basis_bp[:retained].T, index=mats,
                     columns=["PC1", "PC2", "PC3"]).rename_axis("maturity_years").to_csv(output/f"{mode}_raw_bp_basis.csv")
    variance = pd.DataFrame(variance_rows)
    reconstruction = pd.DataFrame(reconstruction_rows)
    variance.to_csv(output/"explained_variance.csv", index=False)
    reconstruction.to_csv(output/"reconstruction.csv", index=False)
    interval, loading_intervals = block_bootstrap(changes, repetitions=bootstrap_repetitions,
                                                 block_length=block_length, seed=seed)
    interval.to_csv(output/"bootstrap_variance.csv")
    loading_intervals.to_csv(output/"bootstrap_loadings.csv", index=False)
    rolling = rolling_pca(changes, window=window)
    rolling.to_csv(output/"rolling_pca.csv")

    # Hypothetical $1m face each: valuation on a coupon date, no accrued interest.
    bonds = [Bond(maturity) for maturity in (2, 5, 10, 30)]
    latest = yields.iloc[-1].to_numpy()
    base_value = price_portfolio(bonds, mats, latest)
    dv01 = key_rate_dv01(bonds, mats, latest)
    model = models["covariance"]
    factor_dv01 = model.basis_bp[:retained] @ dv01
    pd.DataFrame({"maturity_years": mats, "dv01_usd_per_bp": dv01}).to_csv(output/"key_rate_dv01.csv", index=False)
    scenarios = {"parallel_+100bp": np.full(len(mats), 100.),
                 "parallel_-100bp": np.full(len(mats), -100.),
                 "steepener_-25_to_+25bp": np.linspace(-25, 25, len(mats))}
    for j in range(retained):
        one_std = np.sqrt(model.eigenvalues[j]) * model.basis_bp[j]
        scenarios[f"PC{j+1}_+1sigma"] = one_std
        scenarios[f"PC{j+1}_-1sigma"] = -one_std
    scenario_rows = []
    for name, shock in scenarios.items():
        exact = price_portfolio(bonds, mats, latest + shock/100.) - base_value
        linear = float(-shock @ dv01)
        scenario_rows.append({"scenario": name, "full_repricing_pnl_usd": exact,
                              "linear_dv01_pnl_usd": linear,
                              "nonlinear_residual_usd": exact-linear,
                              "full_repricing_return_pct": exact/base_value*100.})
    scenarios_frame = pd.DataFrame(scenario_rows)
    scenarios_frame.to_csv(output/"stress_scenarios.csv", index=False)
    pd.DataFrame(scenarios, index=mats).rename_axis("maturity_years").to_csv(output/"scenario_shocks_bp.csv")

    # Frozen train-sample basis with historical holdout shocks at the latest portfolio curve.
    held_model = trained["covariance"]
    reconstructed = held_model.reconstruct(test.to_numpy(), retained)
    full_linear = -test.to_numpy() @ dv01
    factor_linear = -reconstructed @ dv01
    exact_pnl = np.array([price_portfolio(bonds, mats, latest+shock/100)-base_value
                         for shock in test.to_numpy()])
    pd.DataFrame({"full_linear_pnl_usd": full_linear, "pca_linear_pnl_usd": factor_linear,
                  "full_repricing_pnl_usd": exact_pnl}, index=test.index).to_csv(output/"holdout_static_pnl.csv")
    diagnostics = {"factor_vs_full_linear_rmse_usd": float(np.sqrt(np.mean((factor_linear-full_linear)**2))),
                   "full_linear_vs_repricing_rmse_usd": float(np.sqrt(np.mean((full_linear-exact_pnl)**2)))}
    summary = {"dataset": str(Path(data_path).name),
               "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
               "start": str(yields.index[0].date()), "end": str(yields.index[-1].date()),
               "complete_curves": len(yields), "daily_changes": len(changes),
               "training_changes": len(train), "holdout_changes": len(test),
               "training_end": str(train.index[-1].date()), "holdout_start": str(test.index[0].date()),
               "maturities_years": mats.tolist(),
               "top3_variance_covariance": float(model.explained_variance_ratio[:retained].sum()),
               "top3_variance_correlation": float(models["correlation"].explained_variance_ratio[:retained].sum()),
               "bootstrap_95_interval": interval.iloc[0][["lower_95", "upper_95"]].tolist(),
               "bootstrap_repetitions": bootstrap_repetitions, "block_length": block_length,
               "seed": seed, "rolling_window": window,
               "portfolio_value_usd": base_value, "parallel_dv01_usd": float(dv01.sum()),
               "factor_sensitivity_usd_per_score_unit": factor_dv01.tolist(),
               "pnl_diagnostics": diagnostics,
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}}
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    _plot_results(yields, models, variance, rolling, scenarios_frame, output)
    _write_report(summary, reconstruction, scenarios_frame, output)
    return summary


def _plot_results(yields, models, variance, rolling, scenarios, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .2,
                         "figure.dpi": 150, "font.size": 10})
    palette = ["#193d66", "#237a73", "#c87932"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    yields.loc[:, [1, 2, 5, 10, 30]].plot(ax=axes[0, 0], lw=1.4)
    axes[0, 0].set(title="US Treasury zero-coupon yields", ylabel="Yield (%)", xlabel="")
    axes[0, 0].legend(title="Maturity (years)", ncol=3, fontsize=8)
    for j in range(3):
        axes[0, 1].plot(models["covariance"].maturities, models["covariance"].components[j],
                        marker="o", color=palette[j], label=f"PC{j+1}")
    axes[0, 1].set(title="Covariance PCA loadings", xlabel="Maturity (years)", ylabel="Eigenvector coefficient")
    axes[0, 1].legend()
    for mode in models:
        values = variance[variance["mode"] == mode]
        axes[1, 0].plot(np.arange(1, len(values)+1), values["cumulative_variance"]*100,
                        marker="o", label=mode.title())
    axes[1, 0].set(title="Cumulative variance in each PCA input space", xlabel="Components retained", ylabel="Variance explained (%)")
    axes[1, 0].legend()
    axes[1, 1].plot(rolling.index, rolling["top3_variance"]*100, color=palette[0])
    axes[1, 1].set(title="Trailing-window top-three variance", ylabel="Variance explained (%)", xlabel="")
    axes[1, 1].tick_params(axis="x", rotation=35)
    fig.suptitle(f"Yield-curve PCA | {yields.index[0].date()} to {yields.index[-1].date()}", fontsize=16)
    fig.savefig(output/"research_overview.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    names = scenarios["scenario"].str.replace("_", " ")
    positions = np.arange(len(names))
    ax.barh(positions-.15, scenarios["full_repricing_pnl_usd"]/1000, .3, label="Full repricing", color=palette[0])
    ax.barh(positions+.15, scenarios["linear_dv01_pnl_usd"]/1000, .3, label="Linear DV01", color=palette[1])
    ax.set(yticks=positions, yticklabels=names, xlabel="Portfolio P&L ($ thousands)",
           title="Static hypothetical bond portfolio stress scenarios")
    ax.axvline(0, color="gray", lw=.8)
    ax.legend()
    fig.savefig(output/"stress_scenarios.png")
    plt.close(fig)


def _write_report(summary, reconstruction, scenarios, output):
    s = summary
    lines = ["# Reproducible research results", "",
             f"Data: {s['start']} to {s['end']}; {s['complete_curves']} complete curves and {s['daily_changes']} daily changes.", "",
             f"The first three covariance PCs explain **{s['top3_variance_covariance']:.2%}** of centered raw-bp variance. "
             f"The correlation PCA result is **{s['top3_variance_correlation']:.2%}** of standardized variance; the denominators differ.", "",
             f"The circular block-bootstrap 95% percentile interval for the covariance top-three share is "
             f"**[{s['bootstrap_95_interval'][0]:.2%}, {s['bootstrap_95_interval'][1]:.2%}]** "
             f"using {s['bootstrap_repetitions']} replications, blocks of {s['block_length']} observations, and seed {s['seed']}.", "",
             f"Holdout models were fitted through **{s['training_end']}**. The holdout starts **{s['holdout_start']}** "
             f"and contains {s['holdout_changes']} observed shocks. This evaluates reconstruction of observed changes, not forecasting.", "",
             "## Reconstruction", "", "| Mode | PCs | Sample | RMSE (bp) | Raw-bp energy captured |",
             "|---|---:|---|---:|---:|"]
    for row in reconstruction.itertuples():
        lines.append(f"| {row.mode} | {row.components} | {row.sample} | {row.rmse_bp:.3f} | {row.raw_bp_energy_captured:.2%} |")
    lines += ["", "## Portfolio risk", "",
              "Illustrative regular-coupon bonds: $1m face each at 2, 5, 10, and 30 years; 4% coupons; semiannual payments; no accrued interest.", "",
              f"Portfolio value: **${s['portfolio_value_usd']:,.2f}**. Parallel DV01: **${s['parallel_dv01_usd']:,.2f}/bp**.", "",
              "| Scenario | Full repricing P&L ($) | Linear P&L ($) | Nonlinear residual ($) |",
              "|---|---:|---:|---:|"]
    for row in scenarios.itertuples():
        lines.append(f"| {row.scenario} | {row.full_repricing_pnl_usd:,.2f} | {row.linear_dv01_pnl_usd:,.2f} | {row.nonlinear_residual_usd:,.2f} |")
    lines += ["", "## Interpretation and limits", "",
              "- Level, slope, and curvature are economic interpretations; a given sample need not recover textbook shapes or a fixed ordering.",
              "- Covariance PCA prioritizes maturities with more raw-bp volatility; correlation PCA equalizes training-sample maturity volatility.",
              "- Trailing PCA is a descriptive diagnostic. A fixed initial-window reference is used for cosine matching and subspace angles.",
              "- Block bootstrap preserves dependence within blocks, not long-run nonstationarity. Loading intervals are pointwise and become fragile near tied eigenvalues.",
              "- Bond cashflows are discounted with continuously compounded zero yields and linear interpolation of log discount factors. Cashflows below one year use the origin-to-one-year segment.",
              "- Historical holdout shocks are applied to a fixed latest-date curve and fixed cashflows. Their P&L is a static stress study, not realized returns or a trading backtest.",
              "- These Treasury curves model interest-rate risk, not corporate-credit spreads, defaults, or liquidity risk.",
              "", "## Provenance", "",
              "Data: Federal Reserve Board, Gürkaynak–Sack–Wright nominal yield curve. Current vintages may revise historical observations.",
              "Inspiration: Andrea Ranzato, pca-yield-curve-modelling. This project is an independently coded methodological adaptation, not a numerical reproduction of the dissertation.",
              f"Input SHA-256: `{s['data_sha256']}`.", ""]
    (output/"research_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/gsw_2025.csv")
    parser.add_argument("--output", default="results")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--download", action="store_true", help="Fetch current GSW vintage to --data")
    parser.add_argument("--window", type=int, default=63)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--block-length", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.download:
        download_gsw(args.data)
    summary = run_analysis(args.data, args.output, args.start, args.end, window=args.window,
                           bootstrap_repetitions=args.bootstrap, block_length=args.block_length, seed=args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

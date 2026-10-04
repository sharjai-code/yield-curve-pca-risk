"""Continuous-compounding discount curves and illustrative fixed-rate bonds."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Bond:
    maturity_years: float
    coupon_pct: float = 4.0
    face_value: float = 1_000_000.0
    frequency: int = 2

    def cashflows(self) -> tuple[np.ndarray, np.ndarray]:
        """Regular coupon schedule; valuation falls on a coupon date; no accrual."""
        if self.frequency < 1 or int(self.frequency) != self.frequency:
            raise ValueError("Coupon frequency must be a positive integer.")
        periods = self.maturity_years * self.frequency
        if not np.isfinite([self.maturity_years, self.coupon_pct, self.face_value]).all():
            raise ValueError("Bond inputs must be finite.")
        if periods <= 0 or not np.isclose(periods, round(periods)) or self.face_value <= 0 or self.coupon_pct < 0:
            raise ValueError("Use positive face, nonnegative coupon, and regular maturity.")
        times = np.arange(1, int(round(periods)) + 1) / self.frequency
        amounts = np.full(len(times), self.face_value * self.coupon_pct / 100 / self.frequency)
        amounts[-1] += self.face_value
        return times, amounts


def discount_factors(maturities: np.ndarray, yields_pct: np.ndarray,
                     times: np.ndarray) -> np.ndarray:
    """Interpolate log discount factors with D(0)=1; prohibit long-end extrapolation."""
    mats, yields, t = map(lambda v: np.asarray(v, dtype=float), (maturities, yields_pct, times))
    if mats.ndim != 1 or yields.shape != mats.shape or len(mats) < 2:
        raise ValueError("Require matching maturity and yield vectors.")
    if not np.isfinite(mats).all() or not np.isfinite(yields).all() or not np.isfinite(t).all():
        raise ValueError("Curve and cashflow inputs must be finite.")
    if mats[0] <= 0 or np.any(np.diff(mats) <= 0) or np.any(t < 0) or np.any(t > mats[-1]):
        raise ValueError("Invalid maturity grid or cashflow beyond the curve horizon.")
    log_df = np.r_[0.0, -mats * yields / 100.0]
    return np.exp(np.interp(t, np.r_[0.0, mats], log_df))


def price_portfolio(bonds: list[Bond], maturities: np.ndarray, yields_pct: np.ndarray) -> float:
    total = 0.0
    for bond in bonds:
        times, cash = bond.cashflows()
        total += np.dot(cash, discount_factors(maturities, yields_pct, times))
    return float(total)


def key_rate_dv01(bonds: list[Bond], maturities: np.ndarray, yields_pct: np.ndarray,
                   bump_bp: float = 1.0) -> np.ndarray:
    """Positive dollar sensitivity to a 1 bp fall at each curve node."""
    if bump_bp <= 0 or not np.isfinite(bump_bp):
        raise ValueError("Bump size must be positive and finite.")
    yields = np.asarray(yields_pct, dtype=float)
    out = np.empty(len(yields))
    for j in range(len(yields)):
        bump = np.zeros(len(yields))
        bump[j] = bump_bp / 100.0
        out[j] = (price_portfolio(bonds, maturities, yields-bump) -
                  price_portfolio(bonds, maturities, yields+bump)) / (2*bump_bp)
    return out

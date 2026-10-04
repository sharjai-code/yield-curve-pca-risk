"""PCA with explicit units, training-only scaling, and component alignment."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class PCAModel:
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray  # rows are eigenvectors in PCA input coordinates
    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    maturities: np.ndarray
    mode: str

    @property
    def basis_bp(self) -> np.ndarray:
        """Raw yield-change vector per unit of factor score."""
        return self.components * self.scale[None, :]

    def transform(self, changes: np.ndarray, k: int = 3) -> np.ndarray:
        self._check_k(k)
        values = np.asarray(changes, dtype=float)
        if values.shape[-1] != len(self.mean) or not np.isfinite(values).all():
            raise ValueError("Changes must match training maturities and be finite.")
        return ((values - self.mean) / self.scale) @ self.components[:k].T

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=float)
        k = values.shape[-1]
        self._check_k(k)
        if not np.isfinite(values).all():
            raise ValueError("Factor scores must be finite.")
        return values @ self.basis_bp[:k] + self.mean

    def reconstruct(self, changes: np.ndarray, k: int = 3) -> np.ndarray:
        return self.inverse_transform(self.transform(changes, k))

    def _check_k(self, k: int) -> None:
        if not 1 <= k <= len(self.components):
            raise ValueError("Invalid retained component count.")


def fit_pca(changes: pd.DataFrame | np.ndarray, mode: str = "covariance",
            maturities: np.ndarray | None = None) -> PCAModel:
    """Fit SVD PCA to absolute changes in bp, using sample standard deviations."""
    values = np.asarray(changes, dtype=float)
    if values.ndim != 2 or values.shape[0] <= values.shape[1]:
        raise ValueError("Require a 2-D matrix with more observations than maturities.")
    if not np.isfinite(values).all():
        raise ValueError("PCA input must be finite.")
    if mode not in {"covariance", "correlation"}:
        raise ValueError("Mode must be covariance or correlation.")
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1)
    if np.any(std < 1e-12):
        raise ValueError("A maturity has no variation in the training sample.")
    scale = std if mode == "correlation" else np.ones(values.shape[1])
    normalized = (values - mean) / scale
    _, singular, components = np.linalg.svd(normalized, full_matrices=False)
    eigenvalues = singular ** 2 / (len(values) - 1)
    # Stable display orientation: largest absolute loading is positive.
    signs = np.sign(components[np.arange(len(components)), np.abs(components).argmax(axis=1)])
    components = components * signs[:, None]
    if maturities is None:
        maturities = np.asarray(changes.columns if isinstance(changes, pd.DataFrame)
                                else np.arange(values.shape[1]), dtype=float)
    mats = np.asarray(maturities, dtype=float)
    if mats.shape != (values.shape[1],) or not np.isfinite(mats).all() or np.any(np.diff(mats) <= 0):
        raise ValueError("Maturities must be finite and strictly increasing.")
    return PCAModel(mean, scale, components, eigenvalues, eigenvalues / eigenvalues.sum(), mats, mode)


def align_components(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match eigenvectors by maximum absolute cosine, then align their signs."""
    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    if ref.shape != cand.shape:
        raise ValueError("Aligned bases must have identical dimensions.")
    ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
    cosines = ref @ cand.T
    rows, cols = linear_sum_assignment(-np.abs(cosines))
    order = cols[np.argsort(rows)]
    sign = np.where(cosines[np.arange(len(ref)), order] < 0, -1., 1.)
    return candidate[order] * sign[:, None], order


def block_bootstrap(changes: pd.DataFrame, mode: str = "covariance",
                    repetitions: int = 500, block_length: int = 10,
                    k: int = 3, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Circular moving-block bootstrap. Intervals are pointwise, not simultaneous."""
    reference = fit_pca(changes, mode)
    reference._check_k(k)
    if repetitions < 20 or not 1 <= block_length <= len(changes):
        raise ValueError("Use at least 20 replications and a valid block length.")
    rng = np.random.default_rng(seed)
    values = changes.to_numpy()
    cumulative = np.empty(repetitions)
    aligned = np.empty((repetitions, k, values.shape[1]))
    for b in range(repetitions):
        starts = rng.integers(0, len(values), size=int(np.ceil(len(values) / block_length)))
        indices = ((starts[:, None] + np.arange(block_length)) % len(values)).ravel()[:len(values)]
        model = fit_pca(values[indices], mode, reference.maturities)
        cumulative[b] = model.explained_variance_ratio[:k].sum()
        aligned[b], _ = align_components(reference.components[:k], model.components[:k])
    variance = pd.DataFrame({"estimate": [reference.explained_variance_ratio[:k].sum()],
                             "lower_95": [np.quantile(cumulative, .025)],
                             "upper_95": [np.quantile(cumulative, .975)]}, index=[f"top_{k}"])
    rows = []
    for j in range(k):
        for p, maturity in enumerate(reference.maturities):
            rows.append({"factor": f"PC{j+1}", "maturity_years": maturity,
                         "estimate": reference.components[j, p],
                         "lower_95": np.quantile(aligned[:, j, p], .025),
                         "upper_95": np.quantile(aligned[:, j, p], .975)})
    return variance, pd.DataFrame(rows)


def rolling_pca(changes: pd.DataFrame, window: int = 63, k: int = 3,
                mode: str = "covariance") -> pd.DataFrame:
    """Trailing-window diagnostics, aligned to the first fitted window."""
    if not len(changes.columns) < window <= len(changes):
        raise ValueError("Window must exceed the maturity count and fit the sample.")
    reference = fit_pca(changes.iloc[:window], mode)
    reference._check_k(k)
    rows = []
    for end in range(window, len(changes) + 1):
        model = fit_pca(changes.iloc[end-window:end], mode)
        aligned, order = align_components(reference.components[:k], model.components[:k])
        # Principal angle is invariant to sign and rotations within the retained subspace.
        singular = np.linalg.svd(reference.components[:k] @ model.components[:k].T, compute_uv=False)
        row = {"Date": changes.index[end-1], "top3_variance": model.explained_variance_ratio[:k].sum(),
               "max_subspace_angle_degrees": np.degrees(np.arccos(np.clip(singular.min(), 0, 1)))}
        for j in range(k):
            row[f"PC{j+1}_cosine"] = np.dot(reference.components[j], aligned[j])
            row[f"PC{j+1}_matched_variance"] = model.explained_variance_ratio[order[j]]
        rows.append(row)
    return pd.DataFrame(rows).set_index("Date")

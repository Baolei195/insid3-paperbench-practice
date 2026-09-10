"""Position-debiased cross-image matching on already extracted patch features.

The ordering follows INSID3, Section 3.1, and the author implementation at
0c165a10cf52ab91f335883d06260de86854adbe. No encoder or external data is used.
"""

from __future__ import annotations

from math import fsum

import numpy as np


_EPS = 1e-12


def _feature_grid(value: np.ndarray, name: str) -> np.ndarray:
    """Validate a nonempty channels-last real feature grid without writing to it."""
    array = np.asarray(value)
    if array.ndim != 3 or any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must have nonempty shape (H, W, C)")
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric features")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite float64-representable values")
    return array


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    """Compute x / max(||x||_2, EPS), avoiding overflow for large finite x."""
    scale = np.max(np.abs(values), axis=-1, keepdims=True)
    scaled = values / np.where(scale > 0, scale, 1.0)
    length = np.linalg.norm(scaled, axis=-1, keepdims=True)
    # A nonzero scaled row has length >= 1. An all-zero row stays zero.
    unit = scaled / np.maximum(length, 1.0)
    attenuation = np.minimum((np.minimum(scale, _EPS) / _EPS) * length, 1.0)
    return unit * attenuation


def _stable_mean(values: np.ndarray) -> np.ndarray:
    """Mean of bounded patch rows using accurate, per-channel summation.

    Inputs here are normalized features or bounded error estimates, so
    fsum cannot overflow for any array that fits in memory.
    Unlike a strided reduction, its cancellation error does not accumulate
    with the order in which foreground patches happen to occur.
    """
    return np.array([fsum(column) for column in values.T]) / values.shape[0]


def _row_norm(values: np.ndarray) -> np.ndarray:
    """L2 norms of bounded rows without squaring tiny inputs to zero."""
    scale = np.max(np.abs(values), axis=-1, keepdims=True)
    scaled = values / np.where(scale > 0, scale, 1.0)
    return scale * np.linalg.norm(scaled, axis=-1, keepdims=True)


def _roundoff_scale(values: np.ndarray) -> np.ndarray:
    """Small normwise float64 error estimate for bounded feature rows."""
    return 8 * np.finfo(np.float64).eps * _row_norm(values)


def compute_debiased_similarity(
    probe_features: np.ndarray,
    reference_features: np.ndarray,
    reference_mask: np.ndarray,
    target_features: np.ndarray,
    rank: int,
) -> dict[str, np.ndarray]:
    """Return a positional basis, masked reference prototype and similarity map.

    Args:
        probe_features: Low-semantic features, shape (Hp, Wp, C).
        reference_features: Reference features, shape (Hr, Wr, C).
        reference_mask: Boolean or 0/1 mask, shape (Hr, Wr), with foreground.
        target_features: Target features, shape (Ht, Wt, C).
        rank: Nonnegative integer number of requested singular directions.

    Returns:
        Float64 arrays ``basis`` (C, r), ``reference_prototype`` (C,), and
        ``similarity_map`` (Ht, Wt), with r = min(rank, Hp * Wp, C).

    Raises:
        TypeError: rank is not an integer, or is boolean.
        ValueError: rank is negative, grids are invalid, channel counts differ,
            or the reference mask is invalid or empty.

    Probe patches are normalized, then centered per channel, before SVD.
    Reference and target patches are normalized, projected, and normalized
    again; only then is the foreground mean formed and normalized. Rank zero
    bypasses SVD/projection. Zero singular directions are retained in the
    requested SVD slice, as in the author code. Full-channel projection is
    exactly zero. Projection and foreground-mean cancellation residuals at
    float64 rounding scale are zeroed before normalization. Normwise error
    estimates propagate through patch normalization; means use accurate
    summation. These guards are engineering extensions, not universal SVD
    error bounds. No input is modified.
    """
    if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)):
        raise TypeError("rank must be a nonnegative integer, not a boolean")
    if rank < 0:
        raise ValueError("rank must be nonnegative")

    probe = _feature_grid(probe_features, "probe_features")
    reference = _feature_grid(reference_features, "reference_features")
    target = _feature_grid(target_features, "target_features")
    channels = probe.shape[-1]
    if reference.shape[-1] != channels or target.shape[-1] != channels:
        raise ValueError("all feature grids must have the same channel count")

    mask = np.asarray(reference_mask)
    if mask.shape != reference.shape[:2]:
        raise ValueError("reference_mask must match the reference feature grid")
    if mask.dtype.kind not in "bfiu" or not np.isin(mask, (0, 1)).all():
        raise ValueError("reference_mask must contain only boolean or binary 0/1 values")
    foreground = mask.reshape(-1).astype(bool)
    if not foreground.any():
        raise ValueError("reference_mask must contain at least one foreground patch")

    patch_count = probe.shape[0] * probe.shape[1]
    effective_rank = min(int(rank), patch_count, channels)
    if effective_rank:
        normalized_probe = _normalize_rows(probe.reshape(-1, channels))
        centered_probe = normalized_probe - _stable_mean(normalized_probe)
        _, _, right_vectors = np.linalg.svd(centered_probe, full_matrices=False)
        basis = right_vectors[:effective_rank].T.copy()
    else:
        basis = np.empty((channels, 0), dtype=np.float64)
    absolute_basis = np.abs(basis)

    def project(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        normalized = _normalize_rows(grid.reshape(-1, channels))
        if effective_rank == 0:
            return normalized, _roundoff_scale(normalized)
        if effective_rank == channels:
            return np.zeros_like(normalized), np.zeros((normalized.shape[0], 1))
        residual = normalized - (normalized @ basis) @ basis.T
        # Absolute contraction magnitudes account for cancellation inside
        # both products before estimating a normwise error scale.
        scale = np.abs(normalized) + (
            np.abs(normalized) @ absolute_basis
        ) @ absolute_basis.T
        # A vector norm, rather than per-coordinate cutoffs, keeps a weak
        # signal from disappearing merely because it spans many channels.
        error = _roundoff_scale(scale) + _roundoff_scale(normalized)
        residual_norm = _row_norm(residual)
        numerical_zero = residual_norm <= error
        residual = np.where(numerical_zero, 0.0, residual)
        debiased = _normalize_rows(residual)
        # Normalization can amplify projection error by the reciprocal of a
        # small residual norm. Carry that estimate to foreground aggregation;
        # recomputing it only from unit patches would lose this information.
        error = 2 * error / np.maximum(residual_norm, _EPS) + _roundoff_scale(debiased)
        return debiased, np.where(numerical_zero, 0.0, error)

    reference_debiased, reference_error = project(reference)
    target_debiased, _ = project(target)
    foreground_features = reference_debiased[foreground]
    foreground_mean = _stable_mean(foreground_features)
    mean_error = _stable_mean(reference_error[foreground])[0]
    mean_error += _roundoff_scale(_stable_mean(np.abs(foreground_features))).item()
    mean_norm = _row_norm(foreground_mean).item()
    average_norm = _stable_mean(_row_norm(foreground_features))[0]
    # Limit cancellation cleanup to the length lost through averaging. This
    # leaves aligned weak patches intact, including repeated copies of a
    # single patch whose projection already passed the numerical-zero check.
    cancellation = max(average_norm - mean_norm, 0.0)
    if mean_norm <= min(mean_error, cancellation):
        foreground_mean = np.zeros_like(foreground_mean)
    prototype = _normalize_rows(foreground_mean)
    similarity = (target_debiased @ prototype).reshape(target.shape[:2])
    return {
        "basis": basis,
        "reference_prototype": prototype,
        "similarity_map": similarity,
    }

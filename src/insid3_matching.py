"""Position-debiased cross-image matching on already extracted patch features.

The ordering follows INSID3, Section 3.1, and the author implementation at
0c165a10cf52ab91f335883d06260de86854adbe. No encoder or external data is used.
"""

from __future__ import annotations

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
    attenuation = np.minimum(np.minimum(scale, _EPS) * length / _EPS, 1.0)
    return unit * attenuation


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
    requested SVD slice, as in the author code. No input is modified.
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
        centered_probe = normalized_probe - normalized_probe.mean(axis=0, keepdims=True)
        _, _, right_vectors = np.linalg.svd(centered_probe, full_matrices=False)
        basis = right_vectors[:effective_rank].T.copy()
    else:
        basis = np.empty((channels, 0), dtype=np.float64)

    def project(grid: np.ndarray) -> np.ndarray:
        normalized = _normalize_rows(grid.reshape(-1, channels))
        if effective_rank == 0:
            return normalized
        residual = normalized - (normalized @ basis) @ basis.T
        return _normalize_rows(residual)

    reference_debiased = project(reference)
    target_debiased = project(target)
    prototype = _normalize_rows(reference_debiased[foreground].mean(axis=0))
    similarity = (target_debiased @ prototype).reshape(target.shape[:2])
    return {
        "basis": basis,
        "reference_prototype": prototype,
        "similarity_map": similarity,
    }

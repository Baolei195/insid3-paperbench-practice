"""Analytic and adversarial checks for the position-debiasing mechanism."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from src.insid3_matching import _normalize_rows, compute_debiased_similarity


def sample_inputs():
    return {
        "probe_features": np.array([[[1.0, 0, 0], [-1.0, 0, 0]]]),
        "reference_features": np.array([[[10.0, 1, 0]]]),
        "reference_mask": np.array([[True]]),
        "target_features": np.array([[[-10.0, 1, 0], [10.0, 0, 1]]]),
        "rank": 1,
    }


def test_basis_spans_channel_subspace_and_projection_is_orthogonal():
    # Use a rotated 2D positional subspace, not coordinate-axis-only examples.
    rotation, _ = np.linalg.qr(np.random.default_rng(31).normal(size=(5, 5)))
    u, v, semantic, other, _ = rotation.T
    probe = np.array([u, -u, v, -v, u, -u, v, -v]).reshape(2, 4, 5)
    result = compute_debiased_similarity(
        probe,
        (7 * u + 2 * semantic + other).reshape(1, 1, 5),
        np.ones((1, 1)),
        np.array([semantic - 8 * v, other + 9 * u]).reshape(1, 2, 5),
        2,
    )
    basis = result["basis"]
    projector = basis @ basis.T
    expected_projector = np.outer(u, u) + np.outer(v, v)
    assert basis.shape == (5, 2)
    assert_allclose(basis.T @ basis, np.eye(2), atol=1e-12)
    assert_allclose(projector, expected_projector, atol=1e-12)
    assert_allclose(projector @ projector, projector, atol=1e-12)
    residual = (np.eye(5) - projector) @ (u + semantic)
    assert_allclose(basis.T @ residual, 0, atol=1e-12)
    assert_allclose(result["reference_prototype"], (2 * semantic + other) / np.sqrt(5), atol=1e-12)
    assert_allclose(result["similarity_map"], [[2 / np.sqrt(5), 1 / np.sqrt(5)]], atol=1e-12)


def test_probe_is_centered_per_channel_not_used_as_uncentered_svd():
    inputs = sample_inputs()
    # Constant e2 mean dominates uncentered SVD; spatial variation is solely e1.
    inputs["probe_features"] = np.array([[[0.6, 0.8, 0], [-0.6, 0.8, 0]]])
    basis = compute_debiased_similarity(**inputs)["basis"]
    assert_allclose(basis @ basis.T, np.diag([1, 0, 0]), atol=1e-12)


def test_probe_patches_are_normalized_before_centering():
    inputs = sample_inputs()
    inputs["probe_features"] = np.array([[[100.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]]])
    basis = compute_debiased_similarity(**inputs)["basis"]
    # After normalization there are only e1 and e2; centered variation is e1-e2.
    expected = np.array([[0.5, -0.5, 0], [-0.5, 0.5, 0], [0, 0, 0]])
    assert_allclose(basis @ basis.T, expected, atol=1e-12)


def test_debiasing_reverses_position_driven_false_match():
    inputs = sample_inputs()
    original = compute_debiased_similarity(**{**inputs, "rank": 0})
    debiased = compute_debiased_similarity(**inputs)
    # Same semantics at the opposite position vs unrelated semantics at same position.
    assert_allclose(original["similarity_map"], [[-99 / 101, 100 / 101]])
    assert original["similarity_map"][0, 0] < original["similarity_map"][0, 1]
    assert_allclose(debiased["reference_prototype"], [0, 1, 0], atol=1e-12)
    assert_allclose(debiased["similarity_map"], [[1, 0]], atol=1e-12)
    assert debiased["similarity_map"][0, 0] > debiased["similarity_map"][0, 1]


def test_probe_mean_is_not_subtracted_from_reference_or_target():
    inputs = sample_inputs()
    inputs["probe_features"] = np.array([[[0.6, 0.8, 0], [-0.6, 0.8, 0]]])
    result = compute_debiased_similarity(**inputs)
    assert_allclose(result["reference_prototype"], [0, 1, 0], atol=1e-12)
    assert_allclose(result["similarity_map"], [[1, 0]], atol=1e-12)


def test_normalize_each_projected_patch_before_foreground_mean_then_normalize_mean():
    inputs = sample_inputs()
    inputs["reference_features"] = np.array([[[10.0, 1, 0], [0, 0, 2.0]]])
    inputs["reference_mask"] = np.ones((1, 2))
    inputs["target_features"] = np.array([[[40.0, 3, 0], [-70.0, 0, -2]]])
    result = compute_debiased_similarity(**inputs)
    # Equal directional weight despite unequal magnitudes and removed components.
    assert_allclose(result["reference_prototype"], [0, 1 / np.sqrt(2), 1 / np.sqrt(2)], atol=1e-12)
    assert_allclose(result["similarity_map"], [[1 / np.sqrt(2), -1 / np.sqrt(2)]], atol=1e-12)


def test_changing_mask_changes_concept_but_not_positional_basis():
    inputs = sample_inputs()
    inputs["reference_features"] = np.array([[[4.0, 1, 0], [-8.0, 0, 1]]])
    inputs["target_features"] = np.array([[[7.0, 1, 0], [-3.0, 0, 1], [0, -1, 0]]])
    first = compute_debiased_similarity(**{**inputs, "reference_mask": np.array([[1, 0]])})
    second = compute_debiased_similarity(**{**inputs, "reference_mask": np.array([[0, 1]])})
    assert_allclose(first["basis"] @ first["basis"].T, second["basis"] @ second["basis"].T)
    assert_allclose(first["reference_prototype"], [0, 1, 0], atol=1e-12)
    assert_allclose(second["reference_prototype"], [0, 0, 1], atol=1e-12)
    assert_allclose(first["similarity_map"], [[1, 0, -1]], atol=1e-12)
    assert_allclose(second["similarity_map"], [[0, 1, 0]], atol=1e-12)


def test_rank_zero_is_normalized_matching_without_svd(monkeypatch):
    def forbidden_svd(*args, **kwargs):
        raise AssertionError("rank zero must not perform SVD")

    monkeypatch.setattr(np.linalg, "svd", forbidden_svd)
    result = compute_debiased_similarity(
        np.zeros((1, 2, 2)),
        np.array([[[3.0, 0], [0, 8.0]]]),
        np.ones((1, 2)),
        np.array([[[5.0, 0], [0, -2.0], [1.0, 1.0]]]),
        0,
    )
    assert result["basis"].shape == (2, 0)
    assert_allclose(result["reference_prototype"], np.ones(2) / np.sqrt(2))
    assert_allclose(result["similarity_map"], [[1 / np.sqrt(2), -1 / np.sqrt(2), 1]])


def test_independent_non_square_grids_preserve_row_major_mask_and_target_order():
    reference = np.zeros((2, 3, 3))
    reference[..., 2] = 1
    reference[1, 0] = [10, 1, 0]
    mask = np.zeros((2, 3), dtype=bool)
    mask[1, 0] = True
    target = np.array([
        [[5, 1, 0], [1, 0, 1]],
        [[-4, -1, 0], [8, 1, 1]],
        [[10, 0, 0], [0, 1, -1]],
    ], dtype=float)
    result = compute_debiased_similarity(
        np.array([[[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]]]),
        reference, mask, target, 1,
    )
    assert result["similarity_map"].shape == (3, 2)
    assert_allclose(result["similarity_map"], [[1, 0], [-1, 1 / np.sqrt(2)], [0, 1 / np.sqrt(2)]], atol=1e-12)


@pytest.mark.parametrize("rank,expected_rank", [(0, 0), (1, 1), (2, 2), (100, 2), (np.int64(1), 1)])
def test_rank_is_capped_to_available_svd_columns_not_numerical_rank(rank, expected_rank):
    inputs = sample_inputs()
    # Two centered patches have numerical rank one, but thin SVD provides two columns.
    result = compute_debiased_similarity(**{**inputs, "rank": rank})
    basis = result["basis"]
    assert basis.shape == (3, expected_rank)
    assert_allclose(basis.T @ basis, np.eye(expected_rank), atol=1e-12)


def test_channel_dimension_also_caps_rank_and_full_projection_is_zero():
    probe = np.array([[[1.0, 0], [-1.0, 0], [0, 1.0], [0, -1.0]]])
    result = compute_debiased_similarity(probe, probe, np.ones((1, 4)), probe, 99)
    assert result["basis"].shape == (2, 2)
    assert_allclose(result["basis"] @ result["basis"].T, np.eye(2), atol=1e-12)
    assert_allclose(result["reference_prototype"], 0, atol=1e-12)
    assert_allclose(result["similarity_map"], 0, atol=1e-12)


@pytest.mark.parametrize("probe", [np.zeros((2, 2, 3)), np.ones((1, 1, 3))])
def test_degenerate_probe_still_returns_finite_orthonormal_svd_slice(probe):
    result = compute_debiased_similarity(**{**sample_inputs(), "probe_features": probe})
    assert result["basis"].shape == (3, 1)
    assert_allclose(result["basis"].T @ result["basis"], [[1]], atol=1e-12)
    assert all(np.isfinite(value).all() for value in result.values())


@pytest.mark.parametrize("kind", ["zero_patch", "pure_position", "cancelling_foreground"])
def test_zero_or_cancelling_reference_yields_zero_prototype_and_scores(kind):
    inputs = sample_inputs()
    references = {
        "zero_patch": np.array([[[0.0, 0, 0]]]),
        "pure_position": np.array([[[1.0, 0, 0]]]),
        "cancelling_foreground": np.array([[[1.0, 1, 0], [-1.0, -1, 0]]]),
    }
    inputs["reference_features"] = references[kind]
    inputs["reference_mask"] = np.ones(references[kind].shape[:2])
    result = compute_debiased_similarity(**inputs)
    assert_allclose(result["reference_prototype"], 0, atol=1e-12)
    assert_allclose(result["similarity_map"], 0, atol=1e-12)


def test_nonzero_reference_with_zero_target_has_zero_similarity():
    result = compute_debiased_similarity(**{**sample_inputs(), "target_features": np.zeros((2, 1, 3))})
    assert_allclose(result["reference_prototype"], [0, 1, 0])
    assert_array_equal(result["similarity_map"], np.zeros((2, 1)))


def test_epsilon_does_not_amplify_near_zero_target_to_unit_norm():
    result = compute_debiased_similarity(
        np.zeros((1, 1, 2)), np.array([[[1.0, 0]]]), np.ones((1, 1)),
        np.array([[[1e-15, 0], [0, 0]]]), 0,
    )
    assert_allclose(result["similarity_map"], [[1e-3, 0]], atol=1e-15)


def test_large_finite_features_normalize_without_overflow():
    result = compute_debiased_similarity(
        np.zeros((1, 1, 2)), np.array([[[1e308, 1e308]]]), np.ones((1, 1)),
        np.array([[[1e308, -1e308], [1e308, 1e308]]]), 0,
    )
    assert_allclose(result["similarity_map"], [[0, 1]], atol=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_dtype_conversion_and_inputs_are_not_modified(dtype):
    inputs = sample_inputs()
    for name, value in list(inputs.items()):
        if isinstance(value, np.ndarray):
            inputs[name] = value.astype(dtype)
    before = {key: value.copy() for key, value in inputs.items() if isinstance(value, np.ndarray)}
    result = compute_debiased_similarity(**inputs)
    for key, original in before.items():
        assert_array_equal(inputs[key], original)
        assert all(not np.shares_memory(inputs[key], output) for output in result.values())
    assert all(value.dtype == np.float64 for value in result.values())
    assert_allclose(result["similarity_map"], [[1, 0]], atol=1e-12)


def test_readonly_noncontiguous_views_are_supported_without_mutation():
    inputs = sample_inputs()
    for name, value in list(inputs.items()):
        if isinstance(value, np.ndarray):
            padded = np.repeat(value, 2, axis=1)
            view = padded[:, ::2]
            view.setflags(write=False)
            inputs[name] = view
    before = {key: value.copy() for key, value in inputs.items() if isinstance(value, np.ndarray)}
    result = compute_debiased_similarity(**inputs)
    assert_allclose(result["similarity_map"], [[1, 0]], atol=1e-12)
    for key, original in before.items():
        assert_array_equal(inputs[key], original)
        assert not inputs[key].flags.writeable


@pytest.mark.parametrize("rank", [-1, -10])
def test_negative_rank_is_rejected(rank):
    with pytest.raises(ValueError, match="rank"):
        compute_debiased_similarity(**{**sample_inputs(), "rank": rank})


@pytest.mark.parametrize("rank", [True, np.bool_(False), 1.0, "1", None])
def test_noninteger_or_boolean_rank_is_rejected(rank):
    with pytest.raises(TypeError, match="rank"):
        compute_debiased_similarity(**{**sample_inputs(), "rank": rank})


@pytest.mark.parametrize("field,value,error", [
    ("probe_features", np.ones((2, 3)), "shape"),
    ("reference_features", np.ones((1, 1, 1, 3)), "shape"),
    ("target_features", np.ones((1, 0, 3)), "nonempty"),
    ("probe_features", np.ones((1, 1, 0)), "nonempty"),
    ("reference_features", np.ones((1, 1, 2)), "channel"),
    ("target_features", np.ones((1, 2, 4)), "channel"),
    ("reference_mask", np.ones((1,)), "match"),
    ("reference_mask", np.ones((1, 2)), "match"),
    ("reference_mask", np.zeros((1, 1)), "foreground"),
    ("reference_mask", np.array([[0.5]]), "binary"),
    ("reference_mask", np.array([[np.nan]]), "binary"),
    ("reference_mask", np.array([[2]]), "binary"),
    ("reference_mask", np.array([["1"]]), "binary"),
    ("probe_features", np.full((1, 1, 3), np.nan), "finite"),
    ("reference_features", np.full((1, 1, 3), np.inf), "finite"),
    ("target_features", np.full((1, 1, 3), -np.inf), "finite"),
    ("probe_features", np.ones((1, 1, 3), dtype=complex), "real"),
    ("target_features", np.full((1, 1, 3), "1"), "real"),
])
def test_invalid_inputs_fail_explicitly(field, value, error):
    with pytest.raises(ValueError, match=error):
        compute_debiased_similarity(**{**sample_inputs(), field: value})


def test_rotated_pure_position_cannot_create_a_semantic_reference():
    rotation, _ = np.linalg.qr(np.random.default_rng(5).normal(size=(3, 3)))
    position, semantic, other = rotation.T
    result = compute_debiased_similarity(
        np.array([position, -position]).reshape(1, 2, 3),
        position.reshape(1, 1, 3),
        np.ones((1, 1)),
        np.array([semantic, other, -semantic]).reshape(1, 3, 3),
        1,
    )
    # Normalizing roundoff twice used to create scores as large as 0.89.
    assert_array_equal(result["reference_prototype"], np.zeros(3))
    assert_array_equal(result["similarity_map"], np.zeros((1, 3)))


def test_random_full_channel_projection_has_no_remaining_features():
    rng = np.random.default_rng(19)
    result = compute_debiased_similarity(
        rng.normal(size=(2, 4, 5)),
        rng.normal(size=(2, 3, 5)),
        np.ones((2, 3)),
        rng.normal(size=(3, 2, 5)),
        5,
    )
    assert_allclose(result["basis"] @ result["basis"].T, np.eye(5), atol=1e-12)
    assert_array_equal(result["reference_prototype"], np.zeros(5))
    assert_array_equal(result["similarity_map"], np.zeros((3, 2)))


def test_rotated_semantics_cancel_despite_different_position_components():
    rotation, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
    position, other_position, semantic = rotation.T
    result = compute_debiased_similarity(
        np.array([position, -position, other_position, -other_position]).reshape(2, 2, 3),
        np.array([position + semantic, 2 * position - semantic]).reshape(1, 2, 3),
        np.ones((1, 2)),
        np.array([semantic, -semantic]).reshape(1, 2, 3),
        2,
    )
    # After projection and patch normalization, the foreground is [s, -s].
    assert_array_equal(result["reference_prototype"], np.zeros(3))
    assert_array_equal(result["similarity_map"], np.zeros((1, 2)))


@pytest.mark.parametrize("semantic_strength,expected_score", [(1e-10, 1.0), (1e-13, 0.1)])
def test_real_weak_projected_semantics_survive_roundoff_cleanup(semantic_strength, expected_score):
    result = compute_debiased_similarity(
        np.array([[[1.0, 0, 0], [-1.0, 0, 0]]]),
        np.array([[[0, 1.0, 0]]]),
        np.ones((1, 1)),
        np.array([[[1.0, semantic_strength, 0], [1.0, -semantic_strength, 0]]]),
        1,
    )
    # A residual below normalization EPS is attenuated, not discarded.
    assert_allclose(result["reference_prototype"], [0, 1, 0], atol=1e-15)
    assert_allclose(result["similarity_map"], [[expected_score, -expected_score]], atol=1e-15)


def test_real_weak_foreground_imbalance_is_not_mistaken_for_exact_cancellation():
    result = compute_debiased_similarity(
        np.array([[[1.0, 0, 0], [-1.0, 0, 0]]]),
        np.array([[[0, 1.0, 0], [0, -1.0, 1e-13]]]),
        np.ones((1, 2)),
        np.array([[[0, 0, 1.0], [0, 0, -1.0]]]),
        1,
    )
    # The true mean is 5e-14 e3; epsilon normalization produces 0.05 e3.
    assert_allclose(result["reference_prototype"], [0, 0, 0.05], atol=1e-15)
    assert_allclose(result["similarity_map"], [[0.05, -0.05]], atol=1e-15)


def test_subnormal_row_normalization_preserves_x_over_epsilon():
    tiny = np.nextafter(0.0, 1.0)
    values = np.array([[tiny, tiny], [2 * tiny, -tiny]])
    # Avoid computing a subnormal norm first: all rows are far below EPS.
    assert_allclose(_normalize_rows(values), values / 1e-12, rtol=1e-12, atol=0)


def nondegenerate_inputs():
    rng = np.random.default_rng(246)
    return {
        "probe_features": rng.normal(size=(2, 5, 5)) * np.array([4, 2, 1, 0.5, 0.25]),
        "reference_features": rng.normal(size=(2, 3, 5)),
        "reference_mask": np.array([[1, 0, 1], [0, 1, 0]]),
        "target_features": rng.normal(size=(3, 2, 5)),
        "rank": 2,
    }


def test_matches_independent_channels_first_author_matrix_form():
    inputs = nondegenerate_inputs()
    channels = inputs["probe_features"].shape[-1]

    def normalize_columns(matrix):
        return matrix / np.maximum(np.linalg.norm(matrix, axis=0, keepdims=True), 1e-12)

    # Independently use the author's (C, P) layout, left singular vectors,
    # and explicit orthogonal-complement matrix, rather than the implementation's helper.
    probe = normalize_columns(inputs["probe_features"].reshape(-1, channels).T)
    centered = probe - probe.mean(axis=1, keepdims=True)
    left, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    assert np.all(singular_values[:-1] - singular_values[1:] > 1e-3)
    basis = left[:, :inputs["rank"]]
    complement = np.eye(channels) - basis @ basis.T
    reference = normalize_columns(inputs["reference_features"].reshape(-1, channels).T)
    target = normalize_columns(inputs["target_features"].reshape(-1, channels).T)
    reference = normalize_columns(complement @ reference)
    target = normalize_columns(complement @ target)
    foreground = inputs["reference_mask"].ravel().astype(bool)
    prototype = normalize_columns(reference[:, foreground].mean(axis=1, keepdims=True)).ravel()
    expected_similarity = (prototype @ target).reshape(inputs["target_features"].shape[:2])

    result = compute_debiased_similarity(**inputs)
    assert_allclose(result["basis"] @ result["basis"].T, basis @ basis.T, atol=1e-12)
    assert_allclose(result["reference_prototype"], prototype, atol=1e-12)
    assert_allclose(result["similarity_map"], expected_similarity, atol=1e-12)


def test_matching_is_invariant_to_orthogonal_channel_rotation():
    inputs = nondegenerate_inputs()
    rotation, _ = np.linalg.qr(np.random.default_rng(71).normal(size=(5, 5)))
    rotated_inputs = dict(inputs)
    for name in ("probe_features", "reference_features", "target_features"):
        rotated_inputs[name] = inputs[name] @ rotation
    original = compute_debiased_similarity(**inputs)
    rotated = compute_debiased_similarity(**rotated_inputs)
    original_projector = original["basis"] @ original["basis"].T
    assert_allclose(
        rotated["basis"] @ rotated["basis"].T,
        rotation.T @ original_projector @ rotation,
        atol=1e-12,
    )
    assert_allclose(rotated["reference_prototype"], original["reference_prototype"] @ rotation, atol=1e-12)
    assert_allclose(rotated["similarity_map"], original["similarity_map"], atol=1e-12)


@pytest.mark.parametrize("channels", [3, 768])
def test_zero_channel_padding_preserves_a_real_weak_projected_signal(channels):
    position = np.eye(1, channels, 0).ravel()
    semantic = np.eye(1, channels, 1).ravel()
    result = compute_debiased_similarity(
        np.array([position, -position]).reshape(1, 2, channels),
        semantic.reshape(1, 1, channels),
        np.ones((1, 1)),
        (position + 1e-12 * semantic).reshape(1, 1, channels),
        1,
    )
    # Appending inactive channels cannot turn a resolvable residual into zero.
    assert_allclose(result["reference_prototype"], semantic, atol=1e-15)
    assert_allclose(result["similarity_map"], [[1.0]], atol=1e-15)


@pytest.mark.parametrize("channels", [3, 768])
def test_zero_channel_padding_preserves_a_real_weak_foreground_mean(channels):
    dominant = np.eye(1, channels, 0).ravel()
    semantic = np.eye(1, channels, 1).ravel()
    result = compute_debiased_similarity(
        np.zeros((1, 1, channels)),
        np.array([dominant, -dominant + 1e-12 * semantic]).reshape(1, 2, channels),
        np.ones((1, 2)),
        semantic.reshape(1, 1, channels),
        0,
    )
    # The e1 terms cancel, leaving a mean of 5e-13 e2, then EPS attenuation.
    assert_allclose(result["reference_prototype"], 0.5 * semantic, atol=1e-15)
    assert_allclose(result["similarity_map"], [[0.5]], atol=1e-15)


def test_weak_semantics_spread_across_many_channels_remain_resolvable():
    channels = 768
    position = np.ones(channels) / np.sqrt(channels)
    semantic = np.ones(channels) / np.sqrt(channels)
    semantic[channels // 2:] *= -1
    result = compute_debiased_similarity(
        np.array([position, -position]).reshape(1, 2, channels),
        semantic.reshape(1, 1, channels),
        np.ones((1, 1)),
        (position + 1e-12 * semantic).reshape(1, 1, channels),
        1,
    )
    # The input addition loses a few relative digits, so allow its rounding error.
    assert_allclose(result["similarity_map"], [[1.0]], rtol=0, atol=1e-3)


def test_svd_subspace_estimation_roundoff_cannot_create_a_semantic_reference():
    rng = np.random.default_rng(1)
    positional_basis, _ = np.linalg.qr(rng.normal(size=(32, 3)))
    pure_position = positional_basis @ rng.normal(size=3)
    result = compute_debiased_similarity(
        np.vstack([positional_basis.T, -positional_basis.T]).reshape(2, 3, 32),
        pure_position.reshape(1, 1, 32),
        np.ones((1, 1)),
        rng.normal(size=(2, 3, 32)),
        3,
    )
    # The learned span equals the input span; SVD rounding in small individual
    # coordinates must not turn a fully positional reference into a unit vector.
    assert_array_equal(result["reference_prototype"], np.zeros(32))
    assert_array_equal(result["similarity_map"], np.zeros((2, 3)))

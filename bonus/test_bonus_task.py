"""Analytic checks for a future answer at src/insid3_aggregation.py."""

from pathlib import Path
import importlib

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal


@pytest.fixture(scope="module")
def aggregate():
    answer_path = Path(__file__).resolve().parents[1] / "src" / "insid3_aggregation.py"
    if not answer_path.is_file():
        pytest.fail("Missing model answer: implement src/insid3_aggregation.py::aggregate_from_seed", pytrace=False)
    module = importlib.import_module("src.insid3_aggregation")
    return module.aggregate_from_seed


def case():
    root3 = np.sqrt(3)
    return {
        "original_features": np.array([
            [[3 / 5, 4 / 5, 0], [3 / 5, 4 / 5, 0]],
            [[3 / 5, 2 / 5, 2 * root3 / 5], [3 / 5, 2 / 5, -2 * root3 / 5]],
        ]),
        "debiased_features": np.array([
            [[0, 1, 0], [0, 1, 0]],
            [[0, 1 / 2, root3 / 2], [0, 1 / 2, -root3 / 2]],
        ]),
        "reference_prototype": np.array([0.0, 1, 0]),
        "cluster_labels": np.array([[0, 0], [1, 1]]),
        "candidate_mask": np.array([[True, False], [True, False]]),
        "seed_id": 0,
        "threshold": 0.24,
    }


def test_two_spaces_patch_mean_and_coverage_give_analytic_scores(aggregate):
    inputs = case()
    snapshots = {key: value.copy() for key, value in inputs.items() if isinstance(value, np.ndarray)}
    result = aggregate(**inputs)
    scores = np.asarray(result["cluster_scores"])
    mask = np.asarray(result["mask"])
    assert scores.shape == (2,)
    assert_allclose(scores, [1, 17 / (20 * np.sqrt(13))], atol=1e-7, rtol=1e-6)
    assert mask.dtype == np.bool_
    assert_array_equal(mask, [[True, True], [False, False]])
    for key, snapshot in snapshots.items():
        assert_array_equal(inputs[key], snapshot)


def test_lower_threshold_keeps_entire_cluster_including_noncandidate_patches(aggregate):
    inputs = case()
    inputs["threshold"] = 0.20
    result = aggregate(**inputs)
    assert_allclose(result["cluster_scores"], [1, 17 / (20 * np.sqrt(13))], atol=1e-7, rtol=1e-6)
    assert_array_equal(result["mask"], np.ones((2, 2), dtype=bool))


def test_seed_is_a_label_not_a_hardcoded_zero_or_candidate_pixel_index(aggregate):
    inputs = case()
    inputs["cluster_labels"] = 1 - inputs["cluster_labels"]
    inputs["seed_id"] = 1
    result = aggregate(**inputs)
    assert_allclose(result["cluster_scores"], [17 / (20 * np.sqrt(13)), 1], atol=1e-7, rtol=1e-6)
    assert_array_equal(result["mask"], [[True, True], [False, False]])


def test_threshold_comparison_is_strict_and_does_not_force_seed_inclusion(aggregate):
    # Exact coordinate directions avoid an ambiguous floating-point boundary at 1.
    result = aggregate(
        original_features=np.array([[[1.0, 0], [1.0, 0]]]),
        debiased_features=np.array([[[1.0, 0], [1.0, 0]]]),
        reference_prototype=np.array([1.0, 0]),
        cluster_labels=np.array([[0, 0]]),
        candidate_mask=np.array([[True, False]]),
        seed_id=0,
        threshold=1.0,
    )
    assert_allclose(result["cluster_scores"], [1], atol=1e-7, rtol=1e-6)
    assert_array_equal(result["mask"], [[False, False]])


def test_unequal_clusters_use_all_patches_and_individual_coverage(aggregate):
    # Label 0 has cross similarities 1, 3/5, 3/5, so their mean is 11/15.
    # Its original prototype is (1, 0), intra similarity is 3/5, and coverage
    # is 2/3: score = (11/15) * (3/5) * (2/3) = 22/75.
    # Label 1 has no candidates; seed label 2 has cross score 3/5, not 1.
    features = np.array([
        [[3 / 5, 4 / 5], [3 / 5, 4 / 5], [1, 0],
         [3 / 5, 4 / 5], [3 / 5, -4 / 5], [-1, 0]],
    ])
    result = aggregate(
        original_features=features,
        debiased_features=features.copy(),
        reference_prototype=np.array([1.0, 0]),
        cluster_labels=np.array([[2, 2, 0, 0, 0, 1]]),
        candidate_mask=np.array([[True, False, True, True, False, False]]),
        seed_id=2,
        threshold=0.25,
    )
    assert_allclose(result["cluster_scores"], [22 / 75, 0, 3 / 5], atol=1e-7, rtol=1e-6)
    assert_array_equal(result["mask"], [[True, True, True, True, True, False]])


def test_negative_scores_are_preserved_with_a_negative_threshold(aggregate):
    # The interface supplies a fixed seed and permits any finite threshold.
    # Label 1 has cross similarity -4/5 and intra similarity 3/5.
    features = np.array([[[1.0, 0], [3 / 5, 4 / 5]]])
    result = aggregate(
        original_features=features,
        debiased_features=features.copy(),
        reference_prototype=np.array([0.0, -1]),
        cluster_labels=np.array([[0, 1]]),
        candidate_mask=np.array([[True, True]]),
        seed_id=0,
        threshold=-0.25,
    )
    assert_allclose(result["cluster_scores"], [0, -12 / 25], atol=1e-7, rtol=1e-6)
    assert_array_equal(result["mask"], [[True, False]])


def test_zero_seed_cluster_mean_stays_zero_and_finite(aggregate):
    # Nonzero patches can cancel. A zero seed prototype makes every intra
    # similarity zero, including the seed's similarity to itself.
    features = np.array([[[1.0, 0], [-1.0, 0], [0.0, 1]]])
    result = aggregate(
        original_features=features,
        debiased_features=features.copy(),
        reference_prototype=np.array([0.0, 1]),
        cluster_labels=np.array([[0, 0, 1]]),
        candidate_mask=np.array([[True, False, True]]),
        seed_id=0,
        threshold=0.0,
    )
    scores = np.asarray(result["cluster_scores"])
    assert np.isfinite(scores).all()
    assert_allclose(scores, [0, 0], atol=1e-7, rtol=1e-6)
    assert_array_equal(result["mask"], [[False, False, False]])

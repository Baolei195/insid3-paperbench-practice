"""Analytic checks for a future answer at src/insid3_aggregation.py."""

from pathlib import Path
import importlib.util

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal


@pytest.fixture(scope="module")
def aggregate():
    answer_path = Path(__file__).resolve().parents[1] / "src" / "insid3_aggregation.py"
    if not answer_path.is_file():
        pytest.fail("Missing model answer: implement src/insid3_aggregation.py::aggregate_from_seed", pytrace=False)
    spec = importlib.util.spec_from_file_location("insid3_bonus_answer", answer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
    assert_allclose(scores, [1, 17 / (20 * np.sqrt(13))], atol=1e-12, rtol=1e-12)
    assert mask.dtype == np.bool_
    assert_array_equal(mask, [[True, True], [False, False]])
    for key, snapshot in snapshots.items():
        assert_array_equal(inputs[key], snapshot)


def test_lower_threshold_keeps_entire_cluster_including_noncandidate_patches(aggregate):
    inputs = case()
    inputs["threshold"] = 0.20
    result = aggregate(**inputs)
    assert_allclose(result["cluster_scores"], [1, 17 / (20 * np.sqrt(13))], atol=1e-12, rtol=1e-12)
    assert_array_equal(result["mask"], np.ones((2, 2), dtype=bool))


def test_seed_is_a_label_not_a_hardcoded_zero_or_candidate_pixel_index(aggregate):
    inputs = case()
    inputs["cluster_labels"] = 1 - inputs["cluster_labels"]
    inputs["seed_id"] = 1
    result = aggregate(**inputs)
    assert_allclose(result["cluster_scores"], [17 / (20 * np.sqrt(13)), 1], atol=1e-12, rtol=1e-12)
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
    assert_allclose(result["cluster_scores"], [1], atol=1e-12, rtol=1e-12)
    assert_array_equal(result["mask"], [[False, False]])

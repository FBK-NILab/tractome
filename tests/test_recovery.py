"""Tests for nearest-neighbour fiber recovery (tractome.compute).

These exercise the pure ranking and assignment logic only (numpy +
scikit-learn); no Qt, tractogram IO, or GPU is required.
"""

import numpy as np
import pytest

from tractome.compute import (
    assign_to_nearest_medoid,
    knn_indices,
    nearest_reference,
    rank_recovery_candidates,
)


def _make_embedding(seed=7, n_blobs=40, per_blob=50, dim=40, spread=0.1):
    """Build a synthetic dissimilarity embedding of well-separated blobs.

    Returns the embedding along with the blob id of every row, so tests can
    assert that recovery pulls in siblings rather than arbitrary fibers.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_blobs, dim)) * 10.0
    rows = np.concatenate(
        [centers[i] + rng.normal(size=(per_blob, dim)) * spread for i in range(n_blobs)]
    ).astype(np.float32)
    blob_of = np.repeat(np.arange(n_blobs), per_blob).astype(np.int32)
    return rows, blob_of


@pytest.fixture(scope="module")
def embedding():
    return _make_embedding()


def test_nearest_reference_matches_brute_force():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(200, 8)).astype(np.float32)
    queries = rng.normal(size=(37, 8)).astype(np.float32)

    distances, indices = nearest_reference(reference, queries, chunk=10)

    expected = np.linalg.norm(queries[:, None, :] - reference[None, :, :], axis=2)
    np.testing.assert_array_equal(indices, expected.argmin(axis=1))
    np.testing.assert_allclose(distances, expected.min(axis=1), rtol=1e-5)


def test_nearest_reference_handles_empty_inputs():
    reference = np.zeros((5, 3), dtype=np.float32)
    distances, indices = nearest_reference(reference, np.zeros((0, 3), np.float32))
    assert distances.size == 0
    assert indices.size == 0

    distances, indices = nearest_reference(
        np.zeros((0, 3), np.float32), np.zeros((4, 3), np.float32)
    )
    assert distances.size == 0
    assert indices.size == 0


def test_knn_indices_clamps_k_to_reference_size():
    reference = np.arange(12, dtype=np.float32).reshape(4, 3)
    result = knn_indices(reference, reference, k=99)
    assert result.shape == (4, 4)
    # Every row's own nearest neighbour is itself.
    np.testing.assert_array_equal(result[:, 0], np.arange(4))


def test_recovery_pulls_siblings_of_the_bundle(embedding):
    rows, blob_of = embedding
    bundle_blob = 3
    members = np.flatnonzero(blob_of == bundle_blob)
    bundle_ids, held_out = members[:20], members[20:]
    candidates = np.concatenate([held_out, np.flatnonzero(blob_of != bundle_blob)])

    recovered, distances = rank_recovery_candidates(
        rows, bundle_ids, candidates, budget=len(held_out)
    )

    # Blobs are far apart, so every recovered fiber must be a held-out sibling.
    assert set(recovered.tolist()) == set(held_out.tolist())
    assert np.all(np.diff(distances) >= 0)


def test_recovery_respects_the_budget(embedding):
    rows, blob_of = embedding
    members = np.flatnonzero(blob_of == 0)
    bundle_ids = members[:10]
    candidates = np.flatnonzero(np.isin(np.arange(len(rows)), bundle_ids, invert=True))

    recovered, distances = rank_recovery_candidates(
        rows, bundle_ids, candidates, budget=7
    )

    assert recovered.size == 7
    assert distances.size == 7


def test_recovery_never_returns_bundle_or_excluded_ids(embedding):
    rows, blob_of = embedding
    bundle_ids = np.flatnonzero(blob_of == 5)[:15]
    # Deliberately narrow the pool; recovery must not reach outside it.
    candidates = np.flatnonzero(blob_of == 6)

    recovered, _ = rank_recovery_candidates(rows, bundle_ids, candidates, budget=100)

    assert set(recovered.tolist()).issubset(set(candidates.tolist()))
    assert not set(recovered.tolist()) & set(bundle_ids.tolist())


def test_recovery_is_capped_by_pool_size(embedding):
    rows, blob_of = embedding
    bundle_ids = np.flatnonzero(blob_of == 1)[:5]
    candidates = np.flatnonzero(blob_of == 2)[:3]

    recovered, _ = rank_recovery_candidates(rows, bundle_ids, candidates, budget=1000)

    assert recovered.size == 3


@pytest.mark.parametrize(
    "bundle,candidates,budget",
    [
        ([], [1, 2, 3], 10),
        ([1, 2], [], 10),
        ([1, 2], [3, 4], 0),
    ],
)
def test_recovery_degenerate_inputs_return_empty(embedding, bundle, candidates, budget):
    rows, _ = embedding
    recovered, distances = rank_recovery_candidates(rows, bundle, candidates, budget)
    assert recovered.size == 0
    assert distances.size == 0
    assert recovered.dtype == np.int32


def test_recovery_query_cap_does_not_change_the_answer(embedding):
    """Subsampling a dense bundle must not move which siblings come back."""
    rows, blob_of = embedding
    members = np.flatnonzero(blob_of == 8)
    bundle_ids, held_out = members[:30], members[30:]
    candidates = np.concatenate([held_out, np.flatnonzero(blob_of == 9)])

    full, _ = rank_recovery_candidates(
        rows, bundle_ids, candidates, budget=10, query_cap=0
    )
    capped, _ = rank_recovery_candidates(
        rows, bundle_ids, candidates, budget=10, query_cap=5
    )

    assert set(full.tolist()) == set(capped.tolist())


def test_assignment_picks_the_nearest_medoid(embedding):
    rows, blob_of = embedding
    # One medoid per blob: the first member of each.
    medoid_ids = np.asarray(
        [np.flatnonzero(blob_of == b)[0] for b in range(blob_of.max() + 1)],
        dtype=np.int32,
    )
    to_assign = np.asarray([5, 120, 700, 1300], dtype=np.int32)

    assigned = assign_to_nearest_medoid(rows, to_assign, medoid_ids)

    assert assigned.dtype == np.int32
    # Blobs are well separated, so each fiber lands on its own blob's medoid.
    np.testing.assert_array_equal(blob_of[assigned], blob_of[to_assign])


def test_assignment_returns_cluster_ids_not_positions(embedding):
    rows, _ = embedding
    medoid_ids = np.asarray([300, 900], dtype=np.int32)

    assigned = assign_to_nearest_medoid(rows, np.asarray([301, 901]), medoid_ids)

    assert set(assigned.tolist()).issubset(set(medoid_ids.tolist()))


def test_assignment_handles_empty_inputs(embedding):
    rows, _ = embedding
    assert assign_to_nearest_medoid(rows, [], [1, 2]).size == 0
    assert assign_to_nearest_medoid(rows, [1, 2], []).size == 0
